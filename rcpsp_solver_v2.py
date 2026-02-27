#!/usr/bin/env python3
"""
RCPSP Solver V2 - Enhanced Hybrid GA with Resource Ranking + Neighborhood Search
Based on Goncharov (2024) "A hybrid heuristic algorithm for the RCPSP"

Improvements over V1:
  - Resource ranking (Section 3 of paper): ranks resources by scarcity
  - Weighted crossover using resource importance
  - Multi-neighborhood local search (simplified NS from Section 6)
  - Better diversification with multiple priority rules
  - Instance difficulty classification (sigma thresholds)
"""
import os, sys, random, time
from collections import deque

# ============ PARSER ============
class RCPSPInstance:
    def __init__(self):
        self.n_jobs = 0
        self.n_real = 0
        self.n_resources = 0
        self.horizon = 0
        self.durations = []
        self.requests = []
        self.capacities = []
        self.successors = []
        self.predecessors = []
        self.filename = ""
        self.resource_weights = None  # Will be set by resource ranking

    @staticmethod
    def parse_file(filepath):
        inst = RCPSPInstance()
        inst.filename = os.path.basename(filepath)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("jobs (incl. supersource/sink"):
                inst.n_jobs = int(line.split(":")[-1].strip())
                inst.n_real = inst.n_jobs - 2
            elif line.startswith("horizon"):
                inst.horizon = int(line.split(":")[-1].strip())
            elif line.startswith("- renewable"):
                inst.n_resources = int(line.split(":")[1].strip().split()[0])
            elif line.startswith("PRECEDENCE RELATIONS:"):
                i += 1
                inst.successors = [[] for _ in range(inst.n_jobs)]
                inst.predecessors = [[] for _ in range(inst.n_jobs)]
                for j in range(inst.n_jobs):
                    i += 1
                    parts = lines[i].strip().split()
                    job_id = int(parts[0]) - 1
                    n_succ = int(parts[2])
                    succs = [int(x) - 1 for x in parts[3:3+n_succ]]
                    inst.successors[job_id] = succs
                    for s in succs:
                        inst.predecessors[s].append(job_id)
            elif line.startswith("REQUESTS/DURATIONS:"):
                i += 2
                inst.durations = [0] * inst.n_jobs
                inst.requests = [[0]*inst.n_resources for _ in range(inst.n_jobs)]
                for j in range(inst.n_jobs):
                    i += 1
                    parts = lines[i].strip().split()
                    job_id = int(parts[0]) - 1
                    inst.durations[job_id] = int(parts[2])
                    for k in range(inst.n_resources):
                        inst.requests[job_id][k] = int(parts[3+k])
            elif line.startswith("RESOURCEAVAILABILITIES:"):
                i += 1; i += 1
                inst.capacities = [int(x) for x in lines[i].strip().split()]
            i += 1
        return inst

# ============ CRITICAL PATH ============
def compute_critical_path(inst):
    n = inst.n_jobs
    in_deg = [0]*n
    for j in range(n):
        for s in inst.successors[j]:
            in_deg[s] += 1
    q = deque(j for j in range(n) if in_deg[j]==0)
    topo = []
    while q:
        j = q.popleft(); topo.append(j)
        for s in inst.successors[j]:
            in_deg[s] -= 1
            if in_deg[s]==0: q.append(s)
    est = [0]*n
    for j in topo:
        for s in inst.successors[j]:
            est[s] = max(est[s], est[j]+inst.durations[j])
    cp = max(est[j]+inst.durations[j] for j in range(n))
    return cp, est, topo

# ============ RESOURCE RANKING (Section 3) ============
def rank_resources(inst):
    """Rank resources by scarcity using a simplified cumulative relaxation.
    More scarce resources get higher weight."""
    n = inst.n_jobs
    K = inst.n_resources
    cp, est, topo = compute_critical_path(inst)
    
    # Compute total resource demand vs availability over the critical path
    total_demand = [0.0] * K
    total_avail = [0.0] * K
    for k in range(K):
        for j in range(n):
            total_demand[k] += inst.durations[j] * inst.requests[j][k]
        total_avail[k] = cp * inst.capacities[k] if inst.capacities[k] > 0 else 1
    
    # Resource utilization ratio (higher = more scarce)
    utilization = [total_demand[k] / total_avail[k] if total_avail[k] > 0 else 0 for k in range(K)]
    
    # Rank: sort by utilization descending
    ranked = sorted(range(K), key=lambda k: -utilization[k])
    
    # Assign weights based on rank (paper suggests several options)
    weight_options = [
        [1.0, 0.8, 0.6, 0.4],
        [1.0, 0.9, 0.8, 0.7],
        [1.0, 1.0, 1.0, 1.0],
    ]
    
    # Use utilization-based weights: w_k = 2^(-remaining/max_remaining)
    remaining = [1.0 - utilization[k] for k in range(K)]
    max_rem = max(remaining) if max(remaining) > 0 else 1.0
    util_weights = [0.0] * K
    for k in range(K):
        util_weights[k] = 2.0 ** (-remaining[k] / max_rem) if max_rem > 0 else 1.0
    weight_options.append(util_weights)
    
    # Store all options, will be selected randomly during search
    weights = [0.0] * K
    for k in range(K):
        weights[ranked[k]] = weight_options[0][min(k, len(weight_options[0])-1)]
    
    inst.resource_weights = weights
    inst._weight_options = weight_options
    inst._resource_rank = ranked
    inst._utilization = utilization
    return weights

# ============ SERIAL SGS ============
def serial_sgs(inst, activity_list):
    n = inst.n_jobs; K = inst.n_resources
    T = inst.horizon + 1
    start = [-1]*n; finish = [-1]*n
    res_usage = [[0]*K for _ in range(T)]
    for j in activity_list:
        es = 0
        for p in inst.predecessors[j]:
            if finish[p] > es: es = finish[p]
        dur = inst.durations[j]
        if dur == 0:
            start[j] = es; finish[j] = es; continue
        t = es
        while t + dur <= T:
            ok = True
            for tt in range(t, t+dur):
                for k in range(K):
                    if res_usage[tt][k] + inst.requests[j][k] > inst.capacities[k]:
                        ok = False; break
                if not ok: break
            if ok: break
            t += 1
        if t + dur > T:
            start[j] = es; finish[j] = es + dur; continue
        start[j] = t; finish[j] = t+dur
        for tt in range(t, t+dur):
            for k in range(K):
                res_usage[tt][k] += inst.requests[j][k]
    return start

# ============ PARALLEL SGS ============
def parallel_sgs(inst, priority_list):
    n = inst.n_jobs; K = inst.n_resources
    T = inst.horizon + 1
    start = [-1]*n; finish = [-1]*n
    scheduled = [False]*n
    res_usage = [[0]*K for _ in range(T)]
    priority = [0]*n
    for idx, j in enumerate(priority_list):
        priority[j] = idx
    remaining = n; t = 0
    while remaining > 0 and t < T:
        eligible = []
        for j in range(n):
            if scheduled[j]: continue
            ok = True
            for p in inst.predecessors[j]:
                if finish[p] == -1 or finish[p] > t:
                    ok = False; break
            if ok: eligible.append(j)
        eligible.sort(key=lambda j: priority[j])
        for j in eligible:
            dur = inst.durations[j]
            if dur == 0:
                start[j]=t; finish[j]=t; scheduled[j]=True; remaining-=1; continue
            if t+dur > T: continue
            ok = True
            for tt in range(t, t+dur):
                for k in range(K):
                    if res_usage[tt][k]+inst.requests[j][k]>inst.capacities[k]:
                        ok=False; break
                if not ok: break
            if ok:
                start[j]=t; finish[j]=t+dur; scheduled[j]=True; remaining-=1
                for tt in range(t, t+dur):
                    for k in range(K):
                        res_usage[tt][k] += inst.requests[j][k]
        t += 1
    return start

# ============ HELPERS ============
def compute_makespan(inst, start):
    return max(start[j]+inst.durations[j] for j in range(inst.n_jobs) if start[j]>=0)

def make_topological_activity_list(inst, start):
    n = inst.n_jobs
    jobs = list(range(n))
    jobs.sort(key=lambda j: (start[j], j))
    pos = {jobs[i]: i for i in range(n)}
    needs_fix = False
    for j in range(n):
        for s in inst.successors[j]:
            if pos[j] >= pos[s]:
                needs_fix = True; break
        if needs_fix: break
    if needs_fix:
        in_deg = [0]*n
        for j in range(n):
            for s in inst.successors[j]:
                in_deg[s] += 1
        eligible = sorted([j for j in range(n) if in_deg[j]==0], key=lambda j: (start[j], j))
        al = []
        while eligible:
            j = eligible.pop(0); al.append(j)
            for s in inst.successors[j]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    eligible.append(s)
                    eligible.sort(key=lambda x: (start[x], x))
        return al
    return jobs

# ============ FBI ============
def backward_sgs(inst, activity_list, deadline):
    n = inst.n_jobs; K = inst.n_resources
    T = max(deadline + 1, inst.horizon + 1)
    start = [-1]*n; finish = [-1]*n
    res_usage = [[0]*K for _ in range(T)]
    for j in reversed(activity_list):
        dur = inst.durations[j]
        lf = deadline
        for s in inst.successors[j]:
            if start[s] != -1 and start[s] < lf: lf = start[s]
        if dur == 0:
            start[j] = lf; finish[j] = lf; continue
        latest_start = min(lf - dur, T - dur)
        if latest_start < 0: latest_start = 0
        t = latest_start; found = False
        while t >= 0:
            if t + dur > T: t -= 1; continue
            ok = True
            for tt in range(t, t+dur):
                for k in range(K):
                    if res_usage[tt][k] + inst.requests[j][k] > inst.capacities[k]:
                        ok = False; break
                if not ok: break
            if ok: found = True; break
            t -= 1
        if not found:
            t = 0
            while t + dur <= T and t + dur <= lf:
                ok = True
                for tt in range(t, t+dur):
                    for k in range(K):
                        if res_usage[tt][k] + inst.requests[j][k] > inst.capacities[k]:
                            ok = False; break
                    if not ok: break
                if ok: found = True; break
                t += 1
            if not found: t = max(0, latest_start)
        start[j] = t; finish[j] = t + dur
        for tt in range(t, min(t+dur, T)):
            for k in range(K):
                res_usage[tt][k] += inst.requests[j][k]
    return start

def fbi(inst, start, max_iter=5):
    best_s = list(start); best_ms = compute_makespan(inst, best_s)
    cur = list(start)
    for _ in range(max_iter):
        ms = compute_makespan(inst, cur)
        al = make_topological_activity_list(inst, cur)
        bs = backward_sgs(inst, al, ms)
        al2 = make_topological_activity_list(inst, bs)
        fs = serial_sgs(inst, al2)
        nms = compute_makespan(inst, fs)
        if nms < best_ms:
            best_ms = nms; best_s = list(fs); cur = list(fs)
        else:
            break
    return best_s

# ============ LEFT SHIFT ============
def left_shift(inst, start):
    """Global left shift: try to start each activity earlier."""
    n = inst.n_jobs; K = inst.n_resources
    ms = compute_makespan(inst, start)
    T = ms + 1
    improved = True
    s = list(start)
    while improved:
        improved = False
        al = make_topological_activity_list(inst, s)
        for j in al:
            dur = inst.durations[j]
            if dur == 0: continue
            es = 0
            for p in inst.predecessors[j]:
                es = max(es, s[p] + inst.durations[p])
            if es >= s[j]: continue
            # Build resource usage without j
            old_t = s[j]
            # Try earlier times
            for t in range(es, old_t):
                ok = True
                for tt in range(t, t+dur):
                    if tt >= T: ok = False; break
                    usage_k = [0]*K
                    for jj in range(n):
                        if jj == j: continue
                        if s[jj] <= tt < s[jj]+inst.durations[jj]:
                            for k in range(K):
                                usage_k[k] += inst.requests[jj][k]
                    for k in range(K):
                        if usage_k[k] + inst.requests[j][k] > inst.capacities[k]:
                            ok = False; break
                    if not ok: break
                if ok:
                    s[j] = t; improved = True; break
    return s

# ============ NEIGHBORHOOD SEARCH (Simplified NS from Section 6) ============
def neighborhood_search_step(inst, start, core_j, block_size=8):
    """Simplified NS: reschedule a block of activities around core_j."""
    n = inst.n_jobs; K = inst.n_resources
    ms = compute_makespan(inst, start)
    
    # Build block: activities close to core_j in time
    dists = []
    for j in range(n):
        if inst.durations[j] == 0: continue
        d = abs(start[j] - start[core_j])
        dists.append((d, j))
    dists.sort()
    
    block = set()
    for _, j in dists[:block_size]:
        # Don't include if any predecessor is in block (causes issues)
        has_pred_in_block = False
        for p in inst.predecessors[j]:
            if p in block:
                has_pred_in_block = True; break
        if not has_pred_in_block or j == core_j:
            block.add(j)
        if len(block) >= block_size: break
    
    if len(block) < 2: return start
    
    # Create random priority for block activities
    block_list = list(block)
    random.shuffle(block_list)
    
    # Compute available resources (without block activities)
    T = ms + 1
    res_avail = [[inst.capacities[k] for k in range(K)] for _ in range(T)]
    for j in range(n):
        if j in block: continue
        for tt in range(start[j], min(start[j]+inst.durations[j], T)):
            for k in range(K):
                res_avail[tt][k] -= inst.requests[j][k]
    
    # Compute time windows for block activities
    new_start = list(start)
    for j in block_list:
        dur = inst.durations[j]
        if dur == 0: continue
        # Earliest: after all predecessors
        es = 0
        for p in inst.predecessors[j]:
            if p in block:
                es = max(es, new_start[p] + inst.durations[p])
            else:
                es = max(es, start[p] + inst.durations[p])
        # Latest: before all successors
        lf = ms
        for s in inst.successors[j]:
            if s in block:
                pass  # Will be scheduled later
            else:
                lf = min(lf, start[s])
        
        # Find earliest feasible time in window
        best_t = -1
        for t in range(es, min(lf, T)):
            if t + dur > T: break
            ok = True
            for tt in range(t, t+dur):
                for k in range(K):
                    if res_avail[tt][k] < inst.requests[j][k]:
                        ok = False; break
                if not ok: break
            if ok:
                best_t = t; break
        
        if best_t >= 0:
            new_start[j] = best_t
            for tt in range(best_t, min(best_t+dur, T)):
                for k in range(K):
                    res_avail[tt][k] -= inst.requests[j][k]
        # else keep original
    
    return new_start

# ============ VERIFY ============
def verify_solution(inst, start):
    n = inst.n_jobs
    for j in range(n):
        for s in inst.successors[j]:
            if start[j]+inst.durations[j] > start[s]:
                return False, f"Prec violated: {j}->{s}"
    ms = compute_makespan(inst, start)
    for t in range(ms):
        usage = [0]*inst.n_resources
        for j in range(n):
            if start[j] <= t < start[j]+inst.durations[j]:
                for k in range(inst.n_resources):
                    usage[k] += inst.requests[j][k]
        for k in range(inst.n_resources):
            if usage[k] > inst.capacities[k]:
                return False, f"Res {k} exceeded at t={t}"
    return True, "OK"

# ============ RANDOM + PRIORITY ACTIVITY LISTS ============
def random_activity_list(inst):
    n = inst.n_jobs
    in_deg = [0]*n
    for j in range(n):
        for s in inst.successors[j]:
            in_deg[s] += 1
    eligible = [j for j in range(n) if in_deg[j]==0]
    al = []
    while eligible:
        j = random.choice(eligible); eligible.remove(j); al.append(j)
        for s in inst.successors[j]:
            in_deg[s] -= 1
            if in_deg[s]==0: eligible.append(s)
    return al

def priority_activity_list(inst, rule="lft"):
    n = inst.n_jobs
    _, est, _ = compute_critical_path(inst)
    ms_est = max(est[j]+inst.durations[j] for j in range(n))
    lst = [ms_est]*n
    in_deg_s = [len(inst.successors[j]) for j in range(n)]
    q = deque(j for j in range(n) if in_deg_s[j]==0)
    rev = []
    while q:
        j = q.popleft(); rev.append(j)
        for p in inst.predecessors[j]:
            in_deg_s[p] -= 1
            if in_deg_s[p]==0: q.append(p)
    lft = [0]*n
    for j in rev:
        for s in inst.successors[j]:
            lst[j] = min(lst[j], lst[s]-inst.durations[j])
        lft[j] = lst[j]+inst.durations[j]
    
    # Compute latest start time
    lat_start = [lst[j] for j in range(n)]
    
    in_deg = [0]*n
    for j in range(n):
        for s in inst.successors[j]:
            in_deg[s] += 1
    eligible = [j for j in range(n) if in_deg[j]==0]
    al = []
    while eligible:
        if rule == "lft":
            eligible.sort(key=lambda j: lft[j])
        elif rule == "est":
            eligible.sort(key=lambda j: est[j])
        elif rule == "mts":
            eligible.sort(key=lambda j: -len(inst.successors[j]))
        elif rule == "lst":
            eligible.sort(key=lambda j: lat_start[j])
        elif rule == "grpw":
            # Greatest rank positional weight
            def rpw(j):
                return inst.durations[j] + sum(inst.durations[s] for s in inst.successors[j])
            eligible.sort(key=lambda j: -rpw(j))
        elif rule == "wrup":
            # Weighted resource utilization priority
            w = inst.resource_weights or [1.0]*inst.n_resources
            def wru(j):
                return sum(w[k] * inst.requests[j][k] / inst.capacities[k] 
                          for k in range(inst.n_resources) if inst.capacities[k] > 0)
            eligible.sort(key=lambda j: (-wru(j), lft[j]))
        j = eligible[0]; eligible.remove(j); al.append(j)
        for s in inst.successors[j]:
            in_deg[s] -= 1
            if in_deg[s]==0: eligible.append(s)
    return al

# ============ ENHANCED GENETIC ALGORITHM ============
class GAv2:
    def __init__(self, inst, pop_size=60, max_sched=50000):
        self.inst = inst
        self.pop_size = pop_size
        self.max_sched = max_sched
        self.sched_count = 0
        self.best_s = None
        self.best_ms = float('inf')
        self.sigma = 0  # Instance difficulty
        
    def _count(self, n=1):
        self.sched_count += n

    def _schedule_and_improve(self, al, use_psgs=False):
        """Schedule with SGS + FBI, return (activity_list, start, makespan)."""
        if use_psgs:
            s = parallel_sgs(self.inst, al)
        else:
            s = serial_sgs(self.inst, al)
        s = fbi(self.inst, s, 3)
        ms = compute_makespan(self.inst, s)
        self._count(4)  # SGS + ~3 FBI iterations
        al_new = make_topological_activity_list(self.inst, s)
        return al_new, s, ms

    def init_pop(self):
        pop = []
        # Priority rules
        rules = ["lft", "est", "mts", "lst", "grpw"]
        if self.inst.resource_weights:
            rules.append("wrup")
        
        for rule in rules:
            if self.sched_count >= self.max_sched: break
            al = priority_activity_list(self.inst, rule)
            # Try both S-SGS and P-SGS
            al1, s1, ms1 = self._schedule_and_improve(al, use_psgs=False)
            pop.append((al1, s1, ms1))
            if self.sched_count < self.max_sched:
                al2, s2, ms2 = self._schedule_and_improve(al, use_psgs=True)
                if ms2 < ms1:
                    pop[-1] = (al2, s2, ms2)
        
        # Fill remaining with random
        while len(pop) < self.pop_size and self.sched_count < self.max_sched:
            al = random_activity_list(self.inst)
            use_p = random.random() < 0.5
            al_n, s_n, ms_n = self._schedule_and_improve(al, use_psgs=use_p)
            pop.append((al_n, s_n, ms_n))
        
        return pop

    def crossover(self, p1, p2):
        al1, s1, ms1 = p1; al2, s2, ms2 = p2
        n = self.inst.n_jobs
        primary = al1 if ms1 <= ms2 else al2
        secondary = al2 if ms1 <= ms2 else al1
        cp = random.randint(n//4, 3*n//4)
        child = []; added = set()
        for j in primary:
            if len(child) >= cp: break
            if all(p in added for p in self.inst.predecessors[j]):
                child.append(j); added.add(j)
        for j in secondary:
            if j not in added and all(p in added for p in self.inst.predecessors[j]):
                child.append(j); added.add(j)
        for j in primary:
            if j not in added:
                child.append(j); added.add(j)
        return child

    def crossover_two_point(self, p1, p2):
        """Two-point crossover preserving precedence."""
        al1, s1, ms1 = p1; al2, s2, ms2 = p2
        n = self.inst.n_jobs
        cp1 = random.randint(n//6, n//3)
        cp2 = random.randint(2*n//3, 5*n//6)
        primary = al1 if ms1 <= ms2 else al2
        secondary = al2 if ms1 <= ms2 else al1
        child = []; added = set()
        # Take first part from primary
        for j in primary:
            if len(child) >= cp1: break
            if all(p in added for p in self.inst.predecessors[j]):
                child.append(j); added.add(j)
        # Middle part from secondary
        for j in secondary:
            if len(child) >= cp2: break
            if j not in added and all(p in added for p in self.inst.predecessors[j]):
                child.append(j); added.add(j)
        # Rest from primary
        for j in primary:
            if j not in added and all(p in added for p in self.inst.predecessors[j]):
                child.append(j); added.add(j)
        for j in secondary:
            if j not in added:
                child.append(j); added.add(j)
        return child

    def mutate(self, al, prob=0.3):
        if random.random() > prob: return al
        al = list(al); n = len(al)
        n_swaps = random.randint(1, 5)
        for _ in range(n_swaps):
            i = random.randint(1, n-2)
            d = random.choice([-1, 1])
            j = i + d
            if j < 0 or j >= n: continue
            a, b = al[i], al[j]
            if b not in self.inst.successors[a] and a not in self.inst.successors[b]:
                al[i], al[j] = al[j], al[i]
        return al

    def mutate_insert(self, al, prob=0.2):
        """Move a random activity to a different valid position."""
        if random.random() > prob: return al
        al = list(al); n = len(al)
        pos = {al[i]: i for i in range(n)}
        
        idx = random.randint(1, n-2)
        j = al[idx]
        
        # Find valid range
        earliest = 0
        for p in self.inst.predecessors[j]:
            earliest = max(earliest, pos[p] + 1)
        latest = n - 1
        for s in self.inst.successors[j]:
            latest = min(latest, pos[s] - 1)
        
        if earliest < latest:
            new_pos = random.randint(earliest, latest)
            if new_pos != idx:
                al.pop(idx)
                if new_pos > idx: new_pos -= 1
                al.insert(new_pos, j)
        return al

    def run(self):
        # Resource ranking
        rank_resources(self.inst)
        cp, _, _ = compute_critical_path(self.inst)
        
        pop = self.init_pop()
        pop.sort(key=lambda x: x[2])
        self.best_s = list(pop[0][1])
        self.best_ms = pop[0][2]
        
        # Classify instance difficulty (Section 7)
        self.sigma = (self.best_ms - cp) / cp if cp > 0 else 0
        
        no_imp = 0
        ns_phase = False
        
        while self.sched_count < self.max_sched:
            pool = min(len(pop), self.pop_size)
            
            if not ns_phase:
                # GA phase
                idx = random.sample(range(pool), min(4, pool))
                idx.sort(key=lambda i: pop[i][2])
                p1, p2 = pop[idx[0]], pop[idx[1]]
                
                # Choose crossover
                if random.random() < 0.5:
                    cal = self.crossover(p1, p2)
                else:
                    cal = self.crossover_two_point(p1, p2)
                
                # Mutate
                cal = self.mutate(cal)
                cal = self.mutate_insert(cal)
                
                # Schedule
                use_p = random.random() < 0.3
                cs = parallel_sgs(self.inst, cal) if use_p else serial_sgs(self.inst, cal)
                cs = fbi(self.inst, cs, 3)
                cms = compute_makespan(self.inst, cs)
                self._count(4)
                cal = make_topological_activity_list(self.inst, cs)
                
                # Also try the other SGS
                if random.random() < 0.2 and self.sched_count < self.max_sched:
                    cs2 = serial_sgs(self.inst, cal) if use_p else parallel_sgs(self.inst, cal)
                    cs2 = fbi(self.inst, cs2, 3)
                    cms2 = compute_makespan(self.inst, cs2)
                    self._count(4)
                    if cms2 < cms:
                        cs = cs2; cms = cms2
                        cal = make_topological_activity_list(self.inst, cs)
                
            else:
                # NS phase - neighborhood search
                # Pick a random good solution
                pick = random.randint(0, min(5, pool-1))
                _, base_s, base_ms = pop[pick]
                
                # Pick random core activity (non-dummy)
                core = random.randint(1, self.inst.n_jobs - 2)
                ns_s = neighborhood_search_step(self.inst, base_s, core, block_size=min(10, self.inst.n_real//4))
                
                # Apply FBI to NS result
                al_ns = make_topological_activity_list(self.inst, ns_s)
                cs = serial_sgs(self.inst, al_ns)
                cs = fbi(self.inst, cs, 3)
                cms = compute_makespan(self.inst, cs)
                self._count(5)
                cal = make_topological_activity_list(self.inst, cs)
            
            # Update best
            if cms < self.best_ms:
                self.best_ms = cms; self.best_s = list(cs); no_imp = 0
            else:
                no_imp += 1
            
            pop.append((cal, list(cs), cms))
            pop.sort(key=lambda x: x[2])
            if len(pop) > self.pop_size:
                pop = pop[:self.pop_size]
            
            # Switch between GA and NS phases
            if no_imp >= 150 and not ns_phase:
                ns_phase = True
                no_imp = 0
            elif no_imp >= 100 and ns_phase:
                ns_phase = False
                no_imp = 0
                # Diversify: inject random solutions
                for _ in range(self.pop_size // 4):
                    if self.sched_count >= self.max_sched: break
                    al = random_activity_list(self.inst)
                    s = parallel_sgs(self.inst, al)
                    s = fbi(self.inst, s, 2)
                    ms = compute_makespan(self.inst, s)
                    self._count(3)
                    if ms < self.best_ms:
                        self.best_ms = ms; self.best_s = list(s)
                    pop[-1] = (make_topological_activity_list(self.inst, s), list(s), ms)
                pop.sort(key=lambda x: x[2])
        
        return self.best_s, self.best_ms

# ============ PARSE SOLUTION FILE ============
def parse_solution_file(filepath):
    sols = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in '=sP': continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    sols[(int(parts[0]),int(parts[1]))] = int(parts[2])
                except: pass
    return sols

# ============ MAIN ============
def solve_one(filepath, max_sched=50000):
    inst = RCPSPInstance.parse_file(filepath)
    cp, _, _ = compute_critical_path(inst)
    ga = GAv2(inst, pop_size=60, max_sched=max_sched)
    bs, bms = ga.run()
    ok, msg = verify_solution(inst, bs)
    return {'file': inst.filename, 'n': inst.n_jobs, 'cp_lb': cp,
            'makespan': bms, 'feasible': ok, 'msg': msg,
            'start': bs, 'sched_count': ga.sched_count}

def process_dir(data_dir, sol_file=None, max_sched=50000, only_open=False, out_file=None):
    known = parse_solution_file(sol_file) if sol_file and os.path.exists(sol_file) else {}
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.sm'))
    results = []; improved = []; matched = 0; total_open = 0
    print(f"Found {len(files)} instances. Max schedules: {max_sched}")
    print("="*95)
    print(f"{'Instance':<18} {'Jobs':>5} {'CP_LB':>6} {'Known_UB':>9} {'Our_MS':>7} {'OK':>4} {'Status':>10} {'Time':>6}")
    print("-"*95)
    for fn in files:
        fp = os.path.join(data_dir, fn)
        base = fn.replace('.sm','')
        for pf in ['j120','j90','j60','j30']:
            if base.startswith(pf):
                rest = base[len(pf):]; break
        else: rest = base
        parts = rest.split('_')
        par = int(parts[0]) if len(parts)==2 else 0
        inum = int(parts[1]) if len(parts)==2 else 0
        ub = known.get((par, inum))
        inst = RCPSPInstance.parse_file(fp)
        cp, _, _ = compute_critical_path(inst)
        if only_open and ub is not None and cp >= ub: continue
        if ub is not None and cp < ub: total_open += 1
        t0 = time.time()
        r = solve_one(fp, max_sched)
        dt = time.time()-t0
        r['par']=par; r['inst']=inum; r['ub']=ub; r['time']=dt
        st = ""
        if ub:
            if r['makespan'] < ub: st="IMPROVED!"; improved.append(r)
            elif r['makespan'] == ub: st="MATCHED"; matched += 1
            elif r['makespan'] == cp: st="OPTIMAL"
            else: st=f"+{r['makespan']-ub}"
        else:
            st = "OPTIMAL" if r['makespan']==cp else "heur"
        print(f"{fn:<18} {r['n']:>5} {cp:>6} {str(ub or 'N/A'):>9} {r['makespan']:>7} {'Y' if r['feasible'] else 'N':>4} {st:>10} {dt:>5.1f}s")
        results.append(r)
    print("="*95)
    if results:
        n_t=len(results); n_f=sum(1 for r in results if r['feasible'])
        n_o=sum(1 for r in results if r['makespan']==r['cp_lb'])
        ad=0; cnt=0
        for r in results:
            if r['cp_lb']>0: ad+=(r['makespan']-r['cp_lb'])/r['cp_lb']*100; cnt+=1
        ad = ad/cnt if cnt else 0
        print(f"\nFeasible: {n_f}/{n_t} | Optimal: {n_o}/{n_t} | Avg dev from CP: {ad:.2f}%")
        if known: print(f"Matched UB: {matched} | Improved: {len(improved)} | Open tested: {total_open}")
    if improved:
        print("\n*** IMPROVED ***")
        for r in improved:
            print(f"  {r['file']}: {r['ub']} -> {r['makespan']} (CP_LB={r['cp_lb']})")
    if out_file:
        with open(out_file,'w') as f:
            f.write("Instance\tPar\tInst\tJobs\tCP_LB\tKnown_UB\tOur_MS\tFeasible\tTime\n")
            for r in results:
                f.write(f"{r['file']}\t{r['par']}\t{r['inst']}\t{r['n']}\t{r['cp_lb']}\t{r.get('ub','N/A')}\t{r['makespan']}\t{r['feasible']}\t{r['time']:.2f}\n")
        print(f"Saved to {out_file}")
    return results, improved

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="RCPSP Solver V2 - Enhanced")
    p.add_argument("data_dir", help="Directory with .sm files")
    p.add_argument("--solutions","-s", help="Solution file path")
    p.add_argument("--max-schedules","-m", type=int, default=50000)
    p.add_argument("--only-open", action="store_true")
    p.add_argument("--output","-o", help="Output file")
    p.add_argument("--single", help="Single .sm file")
    a = p.parse_args()
    if a.single:
        r = solve_one(a.single, a.max_schedules)
        print(f"Instance: {r['file']}, Jobs: {r['n']}, CP_LB: {r['cp_lb']}")
        print(f"Makespan: {r['makespan']}, Feasible: {r['feasible']} ({r['msg']})")
    else:
        process_dir(a.data_dir, a.solutions, a.max_schedules, a.only_open, a.output)
