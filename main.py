from envs import eval_env1, eval_env2, eval_env3, eval_env4
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def main():
    dists = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted', 'diagonal']
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
    
    # Выводим места для каждой эвристики по сидам и копим статистику
    ranking_results = {}
    average_places = {}
    for env_name, _ in envs:
        ranking_results[env_name] = []
        place_sums = {dist: 0 for dist in dists}
        print(f'\n{env_name} ranking by seed:')
        for idx, seed in enumerate(seeds):
            ranking_raw = sorted(
                [(dist, data[env_name][dist][idx]) for dist in dists],
                key=lambda x: x[1]
            )
            ranked = []
            prev_value = None
            current_place = 1
            entries_seen = 0
            for dist, value in ranking_raw:
                if prev_value is None:
                    place = 1
                elif value == prev_value:
                    place = current_place
                else:
                    current_place = entries_seen + 1
                    place = current_place
                ranked.append((place, dist, value))
                prev_value = value
                entries_seen += 1
            places = ', '.join([f'{place}) {dist}: {value}' for place, dist, value in ranked])
            print(f'  seed {seed}: {places}')
            
            seed_entry = {
                'seed': seed,
                'ranking': []
            }
            for place, dist, value in ranked:
                seed_entry['ranking'].append({
                    'dist': dist,
                    'place': place,
                    'value': value
                })
                place_sums[dist] += place
            ranking_results[env_name].append(seed_entry)
        
        average_places[env_name] = {
            dist: place_sums[dist] / len(seeds) for dist in dists
        }
        avg_line = ', '.join([f'{dist}: {avg_place:.2f}' for dist, avg_place in average_places[env_name].items()])
        print(f'  Average places: {avg_line}')
    
    # Сохраняем результаты в файл
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_payload = {
        'rankings': ranking_results,
        'average_places': average_places,
        'seeds': seeds,
        'dists': dists
    }
    output_path = os.path.join(results_dir, 'heuristic_rankings.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f'Ranking data saved to {output_path}')
    
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