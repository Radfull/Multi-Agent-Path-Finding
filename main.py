from envs import eval_env1, eval_env2, eval_env3, eval_env4
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from timeit import default_timer as timer

def run_density_experiment():
    # heuristics = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted', 'diagonal']
    heuristics = ['manh', 'diagonal', 'euclid', 'weighted']
    densities = list(range(10, 100, 10))
    seeds = list(range(168, 174))
    envs = [
        ('env1', eval_env1),
        ('env2', eval_env2),
        ('env3', eval_env3),
        ('env4', eval_env4)
    ]
    
    results = {}
    for env_name, env_func in envs:
        results[env_name] = {}
        for density in densities:
            results[env_name][density] = {
                heuristic: {'makespan': [], 'runtime': []} for heuristic in heuristics
            }
            for seed in seeds:
                for heuristic in heuristics:
                    start_time = timer()
                    all_path_lst = env_func(used_dist=heuristic, seed=seed, density_percent=density)
                    elapsed = timer() - start_time
                    makespan = len(all_path_lst)
                    
                    results[env_name][density][heuristic]['makespan'].append(makespan)
                    results[env_name][density][heuristic]['runtime'].append(elapsed)
                    
                    print(f'{env_name} dens={density}% seed={seed} dist={heuristic}: steps={makespan}, runtime={elapsed:.4f}s')
    
    averages = {}
    for env_name in results:
        averages[env_name] = {}
        for density in densities:
            averages[env_name][density] = {}
            for heuristic in heuristics:
                makespan_avg = float(np.mean(results[env_name][density][heuristic]['makespan']))
                runtime_avg = float(np.mean(results[env_name][density][heuristic]['runtime']))
                averages[env_name][density][heuristic] = {
                    'makespan': makespan_avg,
                    'runtime': runtime_avg
                }
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'density_metrics.json')
    payload = {
        'heuristics': heuristics,
        'densities': densities,
        'seeds': seeds,
        'raw_results': results,
        'averages': averages
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'Density metrics saved to {output_path}')
    
    for env_name in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        for heuristic in heuristics:
            makespan_series = [averages[env_name][density][heuristic]['makespan'] for density in densities]
            ax.plot(densities, makespan_series, marker='o', label=heuristic)
        ax.set_xlabel('Environment density (%)', fontsize=12)
        ax.set_ylabel('Average makespan (steps)', fontsize=12)
        ax.set_title(f'{env_name.upper()} - Makespan vs Density', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{env_name}_makespan_vs_density.png', dpi=300, bbox_inches='tight')
        print(f'Saved {env_name}_makespan_vs_density.png')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for heuristic in heuristics:
            runtime_series = [averages[env_name][density][heuristic]['runtime'] for density in densities]
            ax.plot(densities, runtime_series, marker='o', label=heuristic)
        ax.set_xlabel('Environment density (%)', fontsize=12)
        ax.set_ylabel('Average runtime (s)', fontsize=12)
        ax.set_title(f'{env_name.upper()} - Runtime vs Density', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{env_name}_runtime_vs_density.png', dpi=300, bbox_inches='tight')
        print(f'Saved {env_name}_runtime_vs_density.png')
    
    plt.show()


def main():
    run_density_experiment()



if __name__ == "__main__":
    main()