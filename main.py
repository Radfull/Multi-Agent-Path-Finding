from src.envs import eval_env1, eval_env2, eval_env3, eval_env4
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time

start = time.time()
def run_density_experiment(search_type:str='focal', weight:float=1.0):
    '''search_type = ('focal', 'a_star', 'bi_a_star')'''
    heuristics = ['manh', 'diagonal']
    densities = list(range(10, 100, 10))
    seeds = list(range(168, 174))
    # seeds = [169,170]
    envs = [
        # ('env1', eval_env1),
        # # ('env2', eval_env2),
        # ('env3', eval_env3),
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
                    all_path_lst = env_func(search_type=search_type, used_dist=heuristic, seed=seed, density_percent=density, w = weight)
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


def run_a_star_vs_focal_density(weight: float = 1.0, focal_heuristics=None):
    """Строим 4 графика (по окружению) с двумя subplot'ами: makespan и runtime.
    На каждом subplot сравниваем оригинальный A* (манхеттен) и Focal Search с заданными эвристиками."""
    focal_heuristics = focal_heuristics or ['diagonal']
    baseline_algo = ('a_star', 'manh')
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
                'a_star': {'makespan': [], 'runtime': []},
                'focal': {
                    heuristic: {'makespan': [], 'runtime': []}
                    for heuristic in focal_heuristics
                }
            }
            for seed in seeds:
                start_time = timer()
                baseline_paths = env_func(
                    search_type=baseline_algo[0],
                    used_dist=baseline_algo[1],
                    seed=seed,
                    density_percent=density,
                    w=1.0
                )
                elapsed = timer() - start_time
                results[env_name][density]['a_star']['makespan'].append(len(baseline_paths))
                results[env_name][density]['a_star']['runtime'].append(elapsed)
                print(f'{env_name} dens={density}% seed={seed}: A* steps={len(baseline_paths)}, runtime={elapsed:.4f}s')
                
                for heuristic in focal_heuristics:
                    start_time = timer()
                    focal_paths = env_func(
                        search_type='focal',
                        used_dist=heuristic,
                        seed=seed,
                        density_percent=density,
                        w=weight
                    )
                    elapsed = timer() - start_time
                    results[env_name][density]['focal'][heuristic]['makespan'].append(len(focal_paths))
                    results[env_name][density]['focal'][heuristic]['runtime'].append(elapsed)
                    print(f'{env_name} dens={density}% seed={seed}: Focal({heuristic}) steps={len(focal_paths)}, runtime={elapsed:.4f}s')
    
    averages = {}
    for env_name in results:
        averages[env_name] = {}
        for density in densities:
            averages[env_name][density] = {
                'a_star': {
                    'makespan': float(np.mean(results[env_name][density]['a_star']['makespan'])),
                    'runtime': float(np.mean(results[env_name][density]['a_star']['runtime']))
                },
                'focal': {}
            }
            for heuristic in focal_heuristics:
                focal_stats = results[env_name][density]['focal'][heuristic]
                averages[env_name][density]['focal'][heuristic] = {
                    'makespan': float(np.mean(focal_stats['makespan'])),
                    'runtime': float(np.mean(focal_stats['runtime']))
                }
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'a_star_vs_focal_density_metrics.json')
    payload = {
        'baseline': baseline_algo,
        'focal_heuristics': focal_heuristics,
        'densities': densities,
        'seeds': seeds,
        'raw_results': results,
        'averages': averages
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'Density metrics saved to {output_path}')
    
    for env_name, _ in envs:
        name = env_name
        for heuristic in focal_heuristics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            metrics = ['makespan', 'runtime']
            ylabels = ['Average makespan (steps)', 'Average runtime (s)']
            for ax, metric, ylabel in zip(axes, metrics, ylabels):
                baseline_series = [
                    averages[name][density]['a_star'][metric] for density in densities
                ]
                focal_series = [
                    averages[name][density]['focal'][heuristic][metric] for density in densities
                ]
                ax.plot(densities, baseline_series, marker='o', linewidth=2, label='A* (manh)')
                ax.plot(densities, focal_series, marker='o', label=f'Focal ({heuristic})')
                ax.set_title(f'{metric.capitalize()} vs density')
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Environment density (%)')
                ax.grid(True, alpha=0.3)
                ax.legend()
            fig.suptitle(f'{name.upper()} - A* vs Focal ({heuristic})')
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            path = f'{name}_{heuristic}_a_star_vs_focal.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Saved {path}')
    
    plt.show()

'''
def plot_alg_time_steps(env_name: str, seeds: list[int], densities: list[float]):
    """
    Plot comparison of different search algorithms across different densities.
    
    Args:
        env_name: Environment name ('env1', 'env2', 'env3', or 'env4')
        seeds: List of random seeds for averaging results
        densities: List of density values (0.0 to 1.0) to test
    """
    # Select the appropriate eval function based on env_name
    env_map = {
        'env1': eval_env1,
        'env2': eval_env2,
        'env3': eval_env3,
        'env4': eval_env4
    }
    
    if env_name not in env_map:
        raise ValueError(f"Unknown environment name: {env_name}. Must be one of: {list(env_map.keys())}")
    
    eval_func = env_map[env_name]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    
    a_star_t, aw_star_t, focal_t, bi_star_t = [], [], [], []
    a_star, aw_star, focal, bi_star = [], [], [], []
    
    for density in densities:
        a_s, aw_s, f_s, bi_s = 0, 0, 0, 0
        a_t, aw_t, f_t, bi_t = 0.0, 0.0, 0.0, 0.0
        
        for s in seeds:
            a_s += len(eval_func(search_type='a_star', density=density, seed=s))
            aw_s += len(eval_func(search_type='a_star', density=density, seed=s, w=3.0))
            f_s += len(eval_func(search_type='focal', density=density, seed=s))
            bi_s += len(eval_func(search_type='bi_a_star', density=density, seed=s))

            start = timer()
            eval_func(search_type='a_star', density=density, seed=s)
            a_t += timer() - start
            
            start = timer()
            eval_func(search_type='a_star', density=density, seed=s, w=3.0)
            aw_t += timer() - start
            
            start = timer()
            eval_func(search_type='focal', density=density, seed=s)
            f_t += timer() - start
            
            start = timer()
            eval_func(search_type='bi_a_star', density=density, seed=s)
            bi_t += timer() - start
        
        # Average over seeds
        a_star.append(a_s / len(seeds))
        aw_star.append(aw_s / len(seeds))
        focal.append(f_s / len(seeds))
        bi_star.append(bi_s / len(seeds))

        a_star_t.append(a_t / len(seeds))
        aw_star_t.append(aw_t / len(seeds))
        focal_t.append(f_t / len(seeds))
        bi_star_t.append(bi_t / len(seeds))
    
    # Plot time steps
    ax1.plot(densities, a_star, label='A*', marker='o')
    ax1.plot(densities, aw_star, label='AW*', marker='s')
    ax1.plot(densities, focal, label='Focal', marker='^')
    ax1.plot(densities, bi_star, label='Bi A*', marker='d')
    
    ax1.set_ylabel('Time Steps')
    ax1.set_xlabel('Density')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Time Steps vs Density')

    # Plot execution time
    ax2.plot(densities, a_star_t, label='A*', marker='o')
    ax2.plot(densities, aw_star_t, label='AW*', marker='s')
    ax2.plot(densities, focal_t, label='Focal', marker='^')
    ax2.plot(densities, bi_star_t, label='Bi A*', marker='d')
    
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_xlabel('Density')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Execution Time vs Density')

    plt.tight_layout()
    plt.show()
'''



def main():
    # run_density_experiment(search_type='a_star')
    run_a_star_vs_focal_density()
    # plot_alg_time_steps('env2',seeds = [1,2,3,4,5], densities=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # проверить 9 сид
    # dists = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted']
    # best_counts = [0] * 6
    # best_times = [0] * 6
    # print('1 env')
    # start_time = timer()
    
    # for s in range(1,11):
    #     print(str(s) + " / 10")
    #     steps, times = [], []
    #     for d in dists:
    #         st = timer()
    #         steps.append(len(eval_env1(used_dist=d, seed=s, plot=False, search_type='bi_a_star'))) # 1.7 for focal
    #         times.append(round(timer()-st,2))
    #     m = min(steps)
    #     m_t = min(times)
    #     print(m)
    #     for i in range(6):
    #         if steps[i] == m:
    #             best_counts[i] += 1
    #         if times[i] == m_t:
    #             best_times[i] += 1
    
    # print(timer() - start_time)
    # print(best_counts)
    # print(best_times)

    

if __name__ == "__main__":
    main()
    end = time.time()

    print("Время выполнения:", end - start, "сек")