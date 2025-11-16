from envs import eval_env1, eval_env2, eval_env3, eval_env4

def main():
    dists = ['manh', 'euclid', 'cheb','octile', 'mixed', 'weighted']

    print('1 env')
    for d in dists:
        print(f'min_time ({d}): {len(eval_env1(used_dist=d))}')
    
    print('2 env')
    for d in dists:
        print(f'min_time ({d}) : {len(eval_env2(used_dist=d))}')

    print('3 env')
    for d in dists:
        print(f'min_time ({d}): {len(eval_env3(used_dist=d))}')

    print('4 env')
    for d in dists:
        print(f'min_time ({d}): {len(eval_env4(used_dist=d))}')

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