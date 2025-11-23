from src.plot_funcs import plot_alg_time_steps, run_a_star_vs_focal_density


def main():

    # run_density_experiment(search_type='focal')

    plot_alg_time_steps('env1',seeds = [1,2,3, 4, 5], density_percent = [10,20,30,40,50,60,70,80,90])

    # dists = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted', 'diagonal']
    # best_counts = [0] * 6
    # best_times = [0] * 6
    # print('1 env')
    # start_time = timer()
    
    # for s in range(1,11):
    #     print(str(s) + " / 10")
    #     steps, times = [], []
    #     for d in dists:
    #         st = timer()
    #         steps.append(len(eval_env1(used_dist=d, seed=s, plot=False, search_type='a_star', w=1.0))) # 1.7 for focal
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