from src.envs import eval_env1, eval_env2, eval_env3, eval_env4
import time

def main():
    dists = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted']
    best_counts = [0] * 6
    best_times = [0] * 6
    print('1 env')
    start_time = time.time()
    for s in range(1,11):
        print(str(s) + " / 10")
        steps, times = [], []
        for d in dists:
            st = time.time()
            steps.append(len(eval_env1(used_dist=d, seed=s, plot=False)))
            times.append(round(time.time()-st,2))
        m = min(steps)
        m_t = min(times)
        print(m)
        for i in range(6):
            if steps[i] == m:
                best_counts[i] += 1
            if times[i] == m_t:
                best_times[i] += 1
    
    print(time.time() - start_time)
    print(best_counts)
    print(best_times)

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