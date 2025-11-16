from envs import eval_env1, eval_env2, eval_env3, eval_env4
import matplotlib.pyplot as plt
import numpy as np

def main():
    dists = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted']
    seeds = list(range(168, 174))
    envs = [
        ('env1', eval_env1),
        ('env2', eval_env2),
        ('env3', eval_env3),
        ('env4', eval_env4)
    ]
    
    # Собираем данные
    data = {}
    for env_name, env_func in envs:
        data[env_name] = {}
        for d in dists:
            data[env_name][d] = []
            for seed in seeds:
                result = len(env_func(used_dist=d, seed=seed))
                data[env_name][d].append(result)
                print(f'{env_name} seed={seed} dist={d}: {result}')
    
    # Создаем графики
    colors = plt.cm.tab10(np.linspace(0, 1, len(dists)))
    
    for env_name, env_func in envs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, d in enumerate(dists):
            ax.scatter(seeds, data[env_name][d], label=d, color=colors[idx], s=100, alpha=0.7)
        
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('Min Time (steps)', fontsize=12)
        ax.set_title(f'{env_name.upper()} - Distance Metrics Comparison', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(seeds)
        
        plt.tight_layout()
        plt.savefig(f'{env_name}_comparison.png', dpi=300, bbox_inches='tight')
        print(f'Saved {env_name}_comparison.png')
    
    plt.show()

if __name__ == "__main__":
    main()