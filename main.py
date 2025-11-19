from src.envs import eval_env1, eval_env2, eval_env3, eval_env4
import time
import matplotlib.pyplot as plt

def plot_alg_time_steps(env_name: str, seeds: list[int], densities: list[float], y_axis: str = 'steps'):
    """
    Plot comparison of different search algorithms across different densities.
    
    Args:
        env_name: Environment name ('env1', 'env2', 'env3', or 'env4')
        seeds: List of random seeds for averaging results
        densities: List of density values (0.0 to 1.0) to test
        y_axis: Metric for Y-axis - 'steps' for time steps or 'time' for execution time (default: 'steps')
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
    
    if y_axis not in ['steps', 'time']:
        raise ValueError(f"Unknown y_axis: {y_axis}. Must be 'steps' or 'time'")
    
    eval_func = env_map[env_name]
    
    plt.figure(figsize=(12, 7))
    a_star, aw_star, focal = [], [], []
    
    for density in densities:
        a_s, aw_s, f_s = 0, 0, 0
        a_t, aw_t, f_t = 0.0, 0.0, 0.0
        
        for s in seeds:
            if y_axis == 'steps':
                # Measure steps
                a_s += len(eval_func(search_type='a_star', density=density, seed=s))
                aw_s += len(eval_func(search_type='a_star', used_dist='weighted', density=density, seed=s))
                f_s += len(eval_func(search_type='focal', density=density, seed=s))
            else:  # y_axis == 'time'
                # Measure execution time
                start = time.time()
                eval_func(search_type='a_star', density=density, seed=s)
                a_t += time.time() - start
                
                start = time.time()
                eval_func(search_type='a_star', used_dist='weighted', density=density, seed=s)
                aw_t += time.time() - start
                
                start = time.time()
                eval_func(search_type='focal', density=density, seed=s)
                f_t += time.time() - start
        
        # Average over seeds
        if y_axis == 'steps':
            a_star.append(a_s / len(seeds))
            aw_star.append(aw_s / len(seeds))
            focal.append(f_s / len(seeds))
        else:  # y_axis == 'time'
            a_star.append(a_t / len(seeds))
            aw_star.append(aw_t / len(seeds))
            focal.append(f_t / len(seeds))
    
    plt.grid()
    plt.plot(densities, a_star, label='A*', marker='o')
    plt.plot(densities, aw_star, label='AW*', marker='s')
    plt.plot(densities, focal, label='Focal', marker='^')
    plt.legend()
    plt.xlabel('Density')
    
    if y_axis == 'steps':
        plt.ylabel('Time Steps')
        plt.title(f'{env_name.upper()} - Algorithm Time Steps vs Density')
    else:  # y_axis == 'time'
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'{env_name.upper()} - Algorithm Execution Time vs Density')
    
    plt.show()
def main():


    plot_alg_time_steps('env2',seeds = [1,2,3,4,5,6], densities=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], y_axis='time')
    # dists = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted']
    # best_counts = [0] * 6
    # best_times = [0] * 6
    # print('1 env')
    # start_time = time.time()
    # for s in range(1,6):
    #     print(str(s) + " / 10")
    #     steps, times = [], []
    #     for d in dists:
    #         st = time.time()
    #         steps.append(len(eval_env3(used_dist=d, seed=s, plot=False, search_type='focal')))
    #         times.append(round(time.time()-st,2))
    #     m = min(steps)
    #     m_t = min(times)
    #     print(m)
    #     for i in range(6):
    #         if steps[i] == m:
    #             best_counts[i] += 1
    #         if times[i] == m_t:
    #             best_times[i] += 1
    
    # print(time.time() - start_time)
    # print(best_counts)
    # print(best_times)

    # print('1 env') #  для 100 тестов сиды 1-100
    # for d in dists:
    #     print(f'min_time ({d}) : {len(eval_env1(used_dist=d))}')

    # print('2 env') #  для 100 тестов сиды 1-100
    # for d in dists:
    #     print(f'min_time ({d}) : {len(eval_env2(used_dist=d))}')

    # print('3 env') #  для 10 тестов сиды 1-10
    # for d in dists:
    #     print(f'min_time ({d}): {len(eval_env3(used_dist=d))}')

    # print('4 env')
    # for d in dists:
    #     print(f'min_time ({d}): {len(eval_env4(used_dist=d))}')
    
    # Пример использования параметра density (плотность среды):
    # density - это доля от доступных клеток (0.0 до 1.0)
    # density=0.5 означает, что будет использовано 50% доступных клеток для других агентов
    # density=None (по умолчанию) использует стандартные значения
    # 
    # Примеры:
    # eval_env1(density=0.5)  # 50% плотность для env1
    # eval_env2(density=0.8)  # 80% плотность для env2
    # eval_env3(density=0.3)  # 30% плотность для env3
    # eval_env4(density=0.6)  # 60% плотность для env4

# 1 env
# min_time (manh): 27
# min_time (euclid): 27
# min_time (cheb): 28
# min_time (octile): 27
# min_time (mixed): 27
# min_time (weighted): 27
# 2 env
# min_time (manh) : 51
# min_time (euclid) : 43
# min_time (cheb) : 45
# min_time (octile) : 43
# min_time (mixed) : 51
# min_time (weighted) : 43
# 3 env
# min_time (manh): 114
# min_time (euclid): 107
# min_time (cheb): 103
# min_time (octile): 103
# min_time (mixed): 114
# min_time (weighted): 103
# 4 env
# min_time (manh): 143
# min_time (euclid): 134
# min_time (cheb): 131
# min_time (octile): 128
# min_time (mixed): 143
# min_time (weighted): 128

if __name__ == "__main__":
    main()