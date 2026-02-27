#!/usr/bin/env python3
"""
S3C'1447 Challenge - RCPSP Solver Runner
Usage:
    python3 run_challenge.py j60_data -s j60hrs.sm -m 50000 -o results.txt
    python3 run_challenge.py j60_data -s j60hrs.sm -m 50000 -o results.txt --v2
    python3 run_challenge.py j60_data -s j60hrs.sm -m 50000 -o results.txt --all
"""
import os, sys, time, random, platform

def get_machine_info():
    return f"Python {platform.python_version()}, {platform.processor() or platform.machine()}, {platform.system()}"

def main():
    import argparse
    p = argparse.ArgumentParser(description="S3C'1447 Challenge Runner")
    p.add_argument("data_dir", help="Directory with j60 .sm files")
    p.add_argument("--solutions", "-s", default="j60hrs.sm", help="Solution file")
    p.add_argument("--max-schedules", "-m", type=int, default=50000, help="Max schedules")
    p.add_argument("--output", "-o", default="results.txt", help="Output results file")
    p.add_argument("--all", action="store_true", help="Run ALL instances, not just open")
    p.add_argument("--v2", action="store_true", help="Use enhanced V2 solver")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--limit", "-n", type=int, default=0, help="Only test first N open instances (0 = all)")
    a = p.parse_args()
    
    random.seed(a.seed)
    
    if a.v2:
        from rcpsp_solver_v2 import (RCPSPInstance, compute_critical_path, 
            parse_solution_file, solve_one, process_dir)
        print("Using ENHANCED V2 solver (resource ranking + NS)")
    else:
        from rcpsp_solver import (RCPSPInstance, compute_critical_path,
            parse_solution_file, solve_one, process_dir)
        print("Using V1 solver (basic GA + FBI)")
    
    sol_file = a.solutions
    known = parse_solution_file(sol_file) if os.path.exists(sol_file) else {}
    files = sorted(f for f in os.listdir(a.data_dir) if f.endswith('.sm'))
    
    open_instances = []
    closed = 0
    for fn in files:
        fp = os.path.join(a.data_dir, fn)
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
        cp_val = compute_critical_path(inst)[0]
        if ub is not None and cp_val < ub:
            open_instances.append({'file': fn, 'path': fp, 'par': par, 'inst': inum,
                                   'cp_lb': cp_val, 'ub': ub})
        elif ub is not None:
            closed += 1
    
    print(f"\n{'='*95}")
    print(f"S3C'1447 Challenge - RCPSP Solver")
    print(f"{'='*95}")
    print(f"Total instances: {len(files)}")
    print(f"Closed (optimal known): {closed}")
    print(f"Open (LB < UB): {len(open_instances)}")
    print(f"Max schedules: {a.max_schedules}")
    print(f"Machine: {get_machine_info()}")
    print(f"{'='*95}")
    
    if a.all:
        results, improved = process_dir(a.data_dir, sol_file, a.max_schedules, 
                                         only_open=False, out_file=a.output)
    else:
        open_instances.sort(key=lambda x: -(x['ub']-x['cp_lb'])/x['cp_lb'])
        
        if a.limit > 0:
            open_instances = open_instances[:a.limit]
            print(f"\n** Limited to first {a.limit} open instances **")
        
        print(f"\n{'Instance':<16} {'CP_LB':>6} {'Known_UB':>9} {'Gap':>5} {'Gap%':>7} {'Our_MS':>7} {'OK':>4} {'Status':>10} {'Time':>6}")
        print("-"*95)
        
        results = []; improved = []; matched = 0
        for item in open_instances:
            t0 = time.time()
            r = solve_one(item['path'], a.max_schedules)
            dt = time.time() - t0
            r['par'] = item['par']; r['inst'] = item['inst']
            r['ub'] = item['ub']; r['time'] = dt
            gap = item['ub'] - item['cp_lb']
            gap_pct = gap / item['cp_lb'] * 100
            
            if r['makespan'] < item['ub']:
                status = "IMPROVED!"; improved.append(r)
            elif r['makespan'] == item['ub']:
                status = "MATCHED"; matched += 1
            else:
                status = f"+{r['makespan'] - item['ub']}"
            
            print(f"{item['file']:<16} {item['cp_lb']:>6} {item['ub']:>9} {gap:>5} {gap_pct:>6.1f}% {r['makespan']:>7} {'Y' if r['feasible'] else 'N':>4} {status:>10} {dt:>5.1f}s")
            results.append(r)
        
        print("="*95)
        n_f = sum(1 for r in results if r['feasible'])
        ad = 0; cnt = 0
        for r in results:
            if r['cp_lb'] > 0:
                ad += (r['makespan']-r['cp_lb'])/r['cp_lb']*100; cnt += 1
        ad = ad/cnt if cnt else 0
        
        print(f"\nOpen instances tested: {len(results)}")
        print(f"All feasible: {n_f}/{len(results)}")
        print(f"Matched known UB: {matched}")
        print(f"IMPROVED: {len(improved)}")
        print(f"Avg dev from CP: {ad:.2f}%")
        
        if improved:
            print(f"\n{'*'*60}")
            print(f"  IMPROVED SOLUTIONS")
            print(f"{'*'*60}")
            for r in improved:
                print(f"  {r['file']}: UB={r['ub']} -> Our={r['makespan']} (CP={r['cp_lb']})")
        
        if a.output:
            with open(a.output, 'w') as f:
                f.write(f"# S3C'1447 RCPSP Results - {'V2' if a.v2 else 'V1'}\n")
                f.write(f"# Machine: {get_machine_info()}\n")
                f.write(f"# Lambda: {a.max_schedules}\n")
                f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write("Instance\tPar\tInst\tJobs\tCP_LB\tKnown_UB\tOur_MS\tFeasible\tStatus\tTime_sec\n")
                for r in results:
                    ub = r.get('ub','N/A')
                    if ub != 'N/A' and r['makespan'] < ub: st = "IMPROVED"
                    elif ub != 'N/A' and r['makespan'] == ub: st = "MATCHED"
                    elif ub != 'N/A': st = f"+{r['makespan']-ub}"
                    else: st = "heur"
                    f.write(f"{r['file']}\t{r['par']}\t{r['inst']}\t{r['n']}\t{r['cp_lb']}\t{ub}\t{r['makespan']}\t{r['feasible']}\t{st}\t{r['time']:.2f}\n")
            print(f"\nResults saved to {a.output}")

if __name__ == "__main__":
    main()
