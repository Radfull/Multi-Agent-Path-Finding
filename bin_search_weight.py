import time
import json
import os
from statistics import mean
import matplotlib.pyplot as plt


def evaluate_weight(weight: float, env_func, env_name: str, used_dist: str = 'euclid', seeds: list = None, search_type: str = "a_star") -> dict:
    """
    Оценивает вес на одной указанной функции среды и всех сидах.
    Возвращает словарь с результатами: time_steps и computation_time для каждой среды и сида.
    
    Args:
        weight: Вес для оценки
        env_func: Функция среды (например, eval_env1, eval_env52 и т.д.)
        env_name: Имя среды для сохранения результатов (например, 'env1', 'env52')
        used_dist: Используемая метрика расстояния
        seeds: Список сидов для тестирования
    """
    if seeds is None:
        seeds = list(range(168, 172))

    results = {env_name: {}}

    for seed in seeds:
        start_time = time.time()
        try:
            all_path_lst = env_func(used_dist=used_dist, seed=seed, w=weight, search_type=search_type)
            computation_time = time.time() - start_time
            time_steps = len(all_path_lst) if all_path_lst else float('inf')
        except Exception as e:
            computation_time = time.time() - start_time
            time_steps = float('inf')
            print(f"Error in {env_name} seed={seed} weight={weight:.1f}: {e}")

        results[env_name][seed] = {
            'time_steps': time_steps,
            'computation_time': computation_time
        }
        print(f'{env_name} seed={seed} weight={weight:.1f}: steps={time_steps}, time={computation_time:.2f}s')

    return results


def calculate_average_time_steps(results: dict) -> float:
    """
    Вычисляет среднее количество шагов (makespan) по всем средам и сидам.
    """
    all_steps = []
    for env_name in results:
        for seed in results[env_name]:
            steps = results[env_name][seed]['time_steps']
            if steps != float('inf'):
                all_steps.append(steps)

    if len(all_steps) == 0:
        return float('inf')
    return mean(all_steps)


def calculate_average_computation_time(results: dict) -> float:
    """
    Вычисляет среднее время вычисления по всем средам и сидам.
    """
    all_times = []
    for env_name in results:
        for seed in results[env_name]:
            comp_time = results[env_name][seed]['computation_time']
            if comp_time != float('inf'):
                all_times.append(comp_time)

    if len(all_times) == 0:
        return float('inf')
    return mean(all_times)


def _cached_weight_eval(weight: float, env_func, env_name: str, cache: dict, used_dist: str, seeds: list, search_type: str = "a_star") -> dict:
    """
    Helper that evaluates a weight once and caches avg steps + raw results.
    Округляет вес до десятых для корректного кеширования.
    """
    # Округляем вес до десятых для корректного кеширования
    weight_rounded = round(weight, 1)
    
    if weight_rounded in cache:
        print(f"Weight {weight_rounded:.1f}: Average time steps = {cache[weight_rounded]['avg_time_steps']:.2f} (cached)")
        return cache[weight_rounded]

    print(f"\nEvaluating weight = {weight_rounded:.1f}...")
    # Используем округленный вес для вычисления
    results = evaluate_weight(weight_rounded, env_func, env_name, used_dist, seeds, search_type=search_type)
    avg_steps = calculate_average_time_steps(results)
    cache[weight_rounded] = {
        'results': results,
        'avg_time_steps': avg_steps
    }
    print(f"Weight {weight_rounded:.1f}: Average time steps = {avg_steps:.2f}")
    return cache[weight_rounded]


def binary_search_weight(
        env_func,
        env_name: str,
        weight_min: float = 1.0,
        weight_max: float = 5.0,
        used_dist: str = 'euclid',
        seeds: list = None,
        tolerance: float = 0.1,
        max_iterations: int = 20,
        always_eval_weight_1: bool = False,
        search_type: str = "a_star"
) -> dict:
    """
    Бинарный поиск оптимального веса в диапазоне [weight_min, weight_max] для одной функции среды.
    Использует среднее количество шагов как критерий.
    
    Args:
        env_func: Функция среды (например, eval_env1, eval_env52 и т.д.)
        env_name: Имя среды для сохранения результатов
        weight_min: Минимальный вес для поиска
        weight_max: Максимальный вес для поиска
        used_dist: Используемая метрика расстояния
        seeds: Список сидов для тестирования
        tolerance: Точность поиска (по умолчанию 0.1 - до десятых)
        max_iterations: Максимальное количество итераций
        always_eval_weight_1: Если True, всегда вычисляет вес 1.0 для сравнения
        search_type: Тип алгоритма поиска (например, 'a_star', 'focal', 'bi_a_star')
    """
    if seeds is None:
        seeds = list(range(168, 200))

    all_results = {}
    iteration = 0

    print(f"Starting binary search for weight in [{weight_min}, {weight_max}]")
    print(f"Environment function: {env_name}")
    print(f"Using {len(seeds)} seeds: {seeds}")
    print(f"Distance metric: {used_dist}")
    print(f"Search type: {search_type}")
    print(f"Tolerance: {tolerance}")
    if always_eval_weight_1:
        print("Will evaluate weight 1.0 for comparison")
    print("-" * 80)

    # Если нужно, сначала вычисляем вес 1.0 для сравнения
    if always_eval_weight_1 and 1.0 not in all_results:
        weight_1_rounded = round(1.0, 1)
        print(f"\nEvaluating baseline weight = {weight_1_rounded:.1f} for comparison...")
        _cached_weight_eval(weight_1_rounded, env_func, env_name, all_results, used_dist, seeds, search_type)

    while (weight_max - weight_min) > tolerance and iteration < max_iterations:
        iteration += 1
        range_len = weight_max - weight_min
        left_third = weight_min + range_len / 3.0
        right_third = weight_max - range_len / 3.0
        
        # Округляем до десятых
        left_third = round(left_third, 1)
        right_third = round(right_third, 1)

        # Проверяем, что точки не совпадают
        if left_third >= right_third:
            # Если диапазон слишком мал, выходим
            break
            
        print(f"\nIteration {iteration}:")
        print(f"Current range: [{weight_min:.1f}, {weight_max:.1f}]")
        print(f"Testing points: left={left_third:.1f}, right={right_third:.1f}")

        left_result = _cached_weight_eval(left_third, env_func, env_name, all_results, used_dist, seeds, search_type)
        right_result = _cached_weight_eval(right_third, env_func, env_name, all_results, used_dist, seeds, search_type)

        if left_result['avg_time_steps'] <= right_result['avg_time_steps']:
            weight_max = right_third
            print("Left segment has better (or equal) average steps. Shrinking right boundary.")
        else:
            weight_min = left_third
            print("Right segment has better average steps. Shrinking left boundary.")

        print(f"New range: [{weight_min:.1f}, {weight_max:.1f}]")

    # Находим финальный лучший вес из всех протестированных
    if all_results:
        mid_weight = round((weight_min + weight_max) / 2.0, 1)
        if mid_weight not in all_results:
            mid_result = _cached_weight_eval(mid_weight, env_func, env_name, all_results, used_dist, seeds, search_type)
        final_best_weight = min(all_results.keys(), key=lambda w: all_results[w]['avg_time_steps'])
        final_best_steps = all_results[final_best_weight]['avg_time_steps']
    else:
        final_best_weight = 1.0
        final_best_steps = float('inf')

    print("\n" + "=" * 80)
    print(f"Binary search completed after {iteration} iterations")
    print(f"Environment function: {env_name}")
    print(f"Best weight: {final_best_weight:.1f}")
    print(f"Average time steps: {final_best_steps:.2f}")
    if always_eval_weight_1 and 1.0 in all_results:
        weight_1_steps = all_results[1.0]['avg_time_steps']
        print(f"Weight 1.0 (baseline): Average time steps = {weight_1_steps:.2f}")
    print("=" * 80)

    return {
        'env_name': env_name,
        'best_weight': final_best_weight,
        'best_avg_time_steps': final_best_steps,
        'all_results': all_results,
        'iterations': iteration,
        'final_range': [weight_min, weight_max]
    }


def plot_results(weights: list, avg_times: list, avg_makespans: list, env_name: str, used_dist: str, results_dir: str):
    """
    Plots graphs showing the relationship between weight and average computation time/makespan.
    Сохраняет график в файл без показа на экране.
    """
    # Filter out None values
    valid_indices = [i for i, t in enumerate(avg_times) if t is not None and avg_makespans[i] is not None]
    weights_valid = [weights[i] for i in valid_indices]
    avg_times_valid = [avg_times[i] for i in valid_indices]
    avg_makespans_valid = [avg_makespans[i] for i in valid_indices]

    if len(weights_valid) == 0:
        print("\nNo valid data to plot.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')

    # Plot 1: Average computation time vs Weight (scatter plot only, no lines)
    ax1.scatter(weights_valid, avg_times_valid, s=150, c='#2E86AB', marker='o',
                edgecolors='#1B4965', linewidths=2, alpha=0.8, zorder=3)
    ax1.set_xlabel('Weight', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Computation Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Average Computation Time vs Weight\n(Environment: {env_name}, Distance: {used_dist})',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_xlim(min(weights_valid) - 0.3, max(weights_valid) + 0.3)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)

    # Plot 2: Average makespan vs Weight (scatter plot only, no lines)
    ax2.scatter(weights_valid, avg_makespans_valid, s=150, c='#A23B72', marker='o',
                edgecolors='#6B1F3F', linewidths=2, alpha=0.8, zorder=3)
    ax2.set_xlabel('Weight', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Makespan (steps)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Average Makespan vs Weight\n(Environment: {env_name}, Distance: {used_dist})',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_xlim(min(weights_valid) - 0.3, max(weights_valid) + 0.3)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()

    # Save the plot with environment name
    plot_path = os.path.join(results_dir, f'weight_analysis_{env_name}_{used_dist}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nGraphs saved to {plot_path}")
    
    # Закрываем график, чтобы не показывать на экране
    plt.close(fig)


def simple_weight_evaluation(env_func, env_name: str, used_dist: str = 'euclid', seeds: list = None, search_type: str = "a_star"):
    """
    Simple mode: evaluates only weights 1, 2, 3, 4, 5 without binary search.
    """
    if seeds is None:
        seeds = list(range(168, 172))

    print("=" * 80)
    print("Simple Weight Evaluation (weights: 1, 2, 3, 4, 5)")
    print("=" * 80)

    fixed_weights = [1.0, 2.0, 3.0, 4.0, 5.0]

    print(f"\nEvaluating weights: {fixed_weights}")
    print(f"Using {len(seeds)} seeds: {seeds}")
    print(f"Distance metric: {used_dist}")
    print(f"Search type: {search_type}")
    print("-" * 80)

    all_results = {}

    # Evaluate all fixed weights
    for weight in fixed_weights:
        print(f"\nEvaluating weight = {weight:.1f}...")
        results = evaluate_weight(weight, env_func, env_name, used_dist, seeds, search_type)
        avg_steps = calculate_average_time_steps(results)
        avg_time = calculate_average_computation_time(results)
        all_results[weight] = {
            'results': results,
            'avg_time_steps': avg_steps,
            'avg_computation_time': avg_time
        }
        print(f"Weight {weight:.1f}: Average makespan = {avg_steps:.2f}, Average time = {avg_time:.2f}s")

    # Find best weight
    best_weight = min(all_results.keys(), key=lambda w: all_results[w]['avg_time_steps'])
    best_steps = all_results[best_weight]['avg_time_steps']

    return all_results, best_weight, best_steps


def binary_search_mode(env_func, env_name: str, used_dist: str = 'euclid', seeds: list = None, always_eval_weight_1: bool = False, search_type: str = "a_star"):
    """
    Binary search mode: performs binary search for optimal weight for a single environment function.
    
    Args:
        env_func: Функция среды (например, eval_env1, eval_env52 и т.д.)
        env_name: Имя среды для сохранения результатов
        used_dist: Используемая метрика расстояния
        seeds: Список сидов для тестирования
        always_eval_weight_1: Если True, всегда вычисляет вес 1.0 для сравнения
        search_type: Тип алгоритма поиска (например, 'a_star', 'focal', 'bi_a_star')
    """
    if seeds is None:
        seeds = list(range(168, 172))

    print("=" * 80)
    print("Weight Optimization with Binary Search")
    print("=" * 80)
    print(f"Environment function: {env_name}")
    print(f"Using {len(seeds)} seeds: {seeds}")
    print(f"Distance metric: {used_dist}")
    print(f"Search type: {search_type}")
    print("-" * 80)

    # Run binary search
    search_results = binary_search_weight(
        env_func=env_func,
        env_name=env_name,
        weight_min=1.0,
        weight_max=2.0,
        used_dist=used_dist,
        seeds=seeds,
        tolerance=0.1,
        max_iterations=20,
        always_eval_weight_1=always_eval_weight_1,
        search_type=search_type
    )

    # Convert search results to the format we need
    all_results = {}
    for weight, data in search_results['all_results'].items():
        avg_time = calculate_average_computation_time(data['results'])
        all_results[weight] = {
            'results': data['results'],
            'avg_time_steps': data['avg_time_steps'],
            'avg_computation_time': avg_time
        }

    return all_results, search_results['best_weight'], search_results['best_avg_time_steps'], search_results


def save_results(env_name: str, all_results: dict, best_weight: float, best_steps: float, 
                 used_dist: str, seeds: list, results_dir: str, search_results: dict = None):
    """
    Saves results to JSON file. Updates existing file with new environment results.
    """
    output_path = os.path.join(results_dir, 'weight_search_results.json')
    
    # Load existing results if file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            # Если файл поврежден или не может быть прочитан, создаем новый
            output_data = {
                'environments': {},
                'used_dist': used_dist,
                'seeds': seeds
            }
    else:
        output_data = {
            'environments': {},
            'used_dist': used_dist,
            'seeds': seeds
        }
    
    # Убедимся, что ключ 'environments' существует
    if 'environments' not in output_data:
        output_data['environments'] = {}
    
    # Обновим общие параметры, если их нет
    if 'used_dist' not in output_data:
        output_data['used_dist'] = used_dist
    if 'seeds' not in output_data:
        output_data['seeds'] = seeds
    
    # Update or create entry for this environment
    output_data['environments'][env_name] = {
        'best_weight': best_weight,
        'best_avg_time_steps': best_steps,
        'iterations': search_results['iterations'] if search_results else None,
        'final_range': search_results['final_range'] if search_results else None,
        'detailed_results': {}
    }
    
    # Convert results to JSON-compatible format
    for weight, data in all_results.items():
        output_data['environments'][env_name]['detailed_results'][f'{weight:.1f}'] = {
            'avg_time_steps': data['avg_time_steps'],
            'avg_computation_time': data['avg_computation_time'],
            'env_results': {}
        }
        for env in data['results']:
            output_data['environments'][env_name]['detailed_results'][f'{weight:.1f}']['env_results'][env] = {}
            for seed in data['results'][env]:
                output_data['environments'][env_name]['detailed_results'][f'{weight:.1f}']['env_results'][env][str(seed)] = {
                    'time_steps': data['results'][env][seed]['time_steps'],
                    'computation_time': data['results'][env][seed]['computation_time']
                }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")
    return output_path


def _process_single_env(env_func, env_name: str, used_dist: str, seeds: list, results_dir: str, 
                        use_binary_search: bool, always_eval_weight_1: bool, search_type: str = "a_star"):
    """Вспомогательная функция для обработки одной среды."""
    if use_binary_search:
        all_results, best_weight, best_steps, search_results = binary_search_mode(
            env_func, env_name, used_dist, seeds, always_eval_weight_1=always_eval_weight_1, search_type=search_type
        )
    else:
        all_results, best_weight, best_steps = simple_weight_evaluation(env_func, env_name, used_dist, seeds, search_type)
        search_results = None

    # Save results
    save_results(env_name, all_results, best_weight, best_steps, used_dist, seeds, results_dir, search_results)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"Summary Table for {env_name}")
    print("=" * 80)
    print(f"{'Weight':<10} {'Avg Makespan':<20} {'Avg Time (s)':<20}")
    print("-" * 80)

    # Collect data for plots
    weights = []
    avg_times = []
    avg_makespans = []

    # Sort weights for display
    for weight in sorted(all_results.keys()):
        data = all_results[weight]
        weights.append(weight)
        avg_times.append(data['avg_computation_time'] if data['avg_computation_time'] != float('inf') else None)
        avg_makespans.append(data['avg_time_steps'] if data['avg_time_steps'] != float('inf') else None)
        print(f"{weight:<10.1f} {data['avg_time_steps']:<20.2f} {data['avg_computation_time']:<20.2f}")

    # Plot graphs (saves to results directory)
    plot_results(weights, avg_times, avg_makespans, env_name, used_dist, results_dir)
    
    return {
        'env_name': env_name,
        'best_weight': best_weight,
        'best_avg_time_steps': best_steps,
        'all_results': all_results
    }


def main(env_func_or_list, env_name_or_list=None, use_binary_search: bool = True, always_eval_weight_1: bool = False, search_type: str="a_star"):
    """
    Main function - обрабатывает одну или несколько сред.
    
    Args:
        env_func_or_list: Функция среды или список функций (например, eval_env1 или [eval_env1, eval_env2, ...])
        env_name_or_list: Имя среды или список имен (если None, используются имена функций)
        use_binary_search: If True, uses binary search mode. If False, uses simple mode (only weights 1-5).
        always_eval_weight_1: Если True, всегда вычисляет вес 1.0 для сравнения
    """
    used_dist = 'euclid'
    seeds = list(range(168, 200))
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Проверяем, передана одна функция или список
    is_multiple = isinstance(env_func_or_list, list)
    
    if is_multiple:
        # Обработка нескольких сред
        env_functions = env_func_or_list
        if env_name_or_list is None:
            env_names = [func.__name__ for func in env_functions]
        else:
            env_names = env_name_or_list if isinstance(env_name_or_list, list) else [env_name_or_list]
            if len(env_names) != len(env_functions):
                raise ValueError("Количество имен сред должно совпадать с количеством функций")
        
        all_environments_results = {}
        
        print("\n" + "=" * 80)
        print(f"ЗАПУСК ОБРАБОТКИ ДЛЯ {len(env_functions)} СРЕД")
        print("=" * 80)
        
        # Обрабатываем каждую среду по очереди
        for idx, (env_func, env_name) in enumerate(zip(env_functions, env_names), 1):
            print("\n" + "#" * 80)
            print(f"# ОБРАБОТКА СРЕДЫ {idx}/{len(env_functions)}: {env_name}")
            print("#" * 80 + "\n")
            
            try:
                result = _process_single_env(env_func, env_name, used_dist, seeds, results_dir, 
                                            use_binary_search, always_eval_weight_1, search_type)
                all_environments_results[env_name] = result
                
                print(f"\n✓ Среда {env_name} успешно обработана")
                print(f"  Лучший вес: {result['best_weight']:.1f}")
                print(f"  Среднее количество шагов: {result['best_avg_time_steps']:.2f}")
                
            except Exception as e:
                print(f"\n✗ Ошибка при обработке среды {env_name}: {e}")
                import traceback
                traceback.print_exc()
                all_environments_results[env_name] = {
                    'error': str(e),
                    'best_weight': None,
                    'best_avg_time_steps': None
                }
        
        # Выводим общую сводку для всех сред
        print("\n" + "=" * 80)
        print("ОБЩАЯ СВОДКА ПО ВСЕМ СРЕДАМ")
        print("=" * 80)
        print(f"{'Среда':<20} {'Лучший вес':<15} {'Ср. шаги':<15} {'Статус':<15}")
        print("-" * 80)
        
        for env_name in env_names:
            if env_name in all_environments_results:
                result = all_environments_results[env_name]
                if 'error' in result:
                    status = "ОШИБКА"
                    best_weight = "N/A"
                    avg_steps = "N/A"
                else:
                    status = "OK"
                    best_weight = f"{result['best_weight']:.1f}"
                    avg_steps = f"{result['best_avg_time_steps']:.2f}"
                print(f"{env_name:<20} {best_weight:<15} {avg_steps:<15} {status:<15}")
        
        print("=" * 80)
        print(f"\nВсе результаты сохранены в: {results_dir}/weight_search_results.json")
        
        return all_environments_results
    
    else:
        # Обработка одной среды
        env_func = env_func_or_list
        env_name = env_name_or_list if env_name_or_list is not None else env_func.__name__
        
        return _process_single_env(env_func, env_name, used_dist, seeds, results_dir, 
                                   use_binary_search, always_eval_weight_1)


if __name__ == "__main__":
    # ============================================
    # ИЗМЕНИТЕ ЭТО: Укажите функции сред
    # ============================================
    
    from src.envs import eval_env1, eval_env2, eval_env3, eval_env4
    
    # ВАРИАНТ 1: ОДНА СРЕДА
    # ENV = eval_env3  # Просто укажите функцию
    
    # ВАРИАНТ 2: НЕСКОЛЬКО СРЕД
    ENV = [
        eval_env1,
        eval_env2,
        eval_env3,
        eval_env4,
    ]
    
    # Имена сред (опционально, если None - используются имена функций)
    ENV_NAMES = None  # Например: ['env1', 'env2', 'env3', ...] или None
    
    # Режим работы: True для бинарного поиска, False для простого режима (только веса 1-5)
    USE_BINARY_SEARCH = True
    
    # Всегда вычислять вес 1.0 для сравнения
    ALWAYS_EVAL_WEIGHT_1 = True
    
    # Тип алгоритма поиска: 'a_star', 'focal', 'bi_a_star', 'lazy_a_star' и т.д.
    SEARCH_TYPE = "bi_a_star"  # Измените на нужный тип алгоритма
    
    # ============================================
    
    main(env_func_or_list=ENV, env_name_or_list=ENV_NAMES, 
         use_binary_search=USE_BINARY_SEARCH, always_eval_weight_1=ALWAYS_EVAL_WEIGHT_1, search_type=SEARCH_TYPE)

