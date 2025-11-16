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
# min_time (euclid): 37
# min_time (cheb): 24
# min_time (octile): 20
# min_time (mixed): 18
# min_time (weighted): 15
# 2 env
# min_time (manh) : 11
# min_time (euclid) : 19
# min_time (cheb) : 30
# min_time (octile) : 36
# min_time (mixed) : 29
# min_time (weighted) : 22
# 3 env
# min_time (manh): 137
# min_time (euclid): 121
# min_time (cheb): 118
# min_time (octile): 114
# min_time (mixed): 136
# min_time (weighted): 114
# 4 env
# min_time (manh): 140
# min_time (euclid): 92
# min_time (cheb): 105
# min_time (octile): 116
# min_time (mixed): 132
# min_time (weighted): 130

if __name__ == "__main__":
    main()