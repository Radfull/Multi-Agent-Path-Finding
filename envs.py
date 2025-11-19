import random
from solver.phans_null_agent_swapping import PHANS


def _compute_other_agents(
    size_x: int,
    size_y: int,
    density_percent: int,
    static_obstacles_count: int = 0,
    density_boost: int = 0,
    max_available: int | None = None,
) -> int:
    adjusted_density = max(0, density_percent + density_boost)
    total_slots = int(size_x * size_y * adjusted_density / 100)
    num_agents = max(0, total_slots - static_obstacles_count)
    if max_available is not None:
        num_agents = min(num_agents, max_available)
    return num_agents


def eval_env1(plot:bool = False, used_dist:str = 'manh', seed = 168, density_percent: int = 90):                
    # --------------------------
    # problem definition
    # --------------------------
    random.seed(seed)
    size_x = 14
    size_y = 7
    
    static_obss = []

    task_lst = [(0, 0), (size_x-1, 0)]    
    
    tgt_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+task_lst))
    tgt_ag_start_pos_lst = random.sample(tgt_ag_option_nodes, len(task_lst))

    other_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+tgt_ag_start_pos_lst))
    max_available_nodes = len(other_ag_option_nodes)
    num_other_agents = _compute_other_agents(size_x, size_y, density_percent, max_available=max_available_nodes)
    other_ag_start_pos_lst = random.sample(other_ag_option_nodes, num_other_agents)

    ag_start_pos_lst = tgt_ag_start_pos_lst + other_ag_start_pos_lst

    # --------------------------
    # Run
    # --------------------------
    problem = PHANS(size_x, size_y, static_obstacles=static_obss, used_dist = used_dist)
    all_path_lst = problem.run_loop(ag_start_pos_lst, task_lst)
    
    # --------------------------
    # Save animation
    # --------------------------
    if plot:
        animation_name = f'{size_x}x{size_y}_{len(static_obss)}obs_{len(ag_start_pos_lst)}ags_{len(task_lst)}tgts_{len(all_path_lst)}steps.gif'
        problem.plot_animation(animation_name, all_path_lst, small_env=True)

    return all_path_lst

def eval_env2(plot:bool = False, used_dist:str = 'manh', seed = 168, density_percent: int = 90):                
    # --------------------------
    # problem definition
    # --------------------------
    random.seed(seed)
    size_x = 14
    size_y = 7
    
    static_obss = [(2, 2), (2, 3), (2, 4), 
                   (3, 2), (3, 3), (3, 4),  
                   (10, 2), (10, 3), (10, 4), 
                   (11, 2), (11, 3), (11, 4)]

    task_lst = [(0, 0), (size_x-1, 0)]    
    
    tgt_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+task_lst))
    tgt_ag_start_pos_lst = random.sample(tgt_ag_option_nodes, len(task_lst))

    other_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+tgt_ag_start_pos_lst))
    max_available_nodes = len(other_ag_option_nodes)
    num_other_agents = _compute_other_agents(
        size_x,
        size_y,
        density_percent,
        static_obstacles_count=len(static_obss),
        max_available=max_available_nodes,
    )
    other_ag_start_pos_lst = random.sample(other_ag_option_nodes, num_other_agents)

    ag_start_pos_lst = tgt_ag_start_pos_lst + other_ag_start_pos_lst

    # --------------------------
    # Run
    # --------------------------
    problem = PHANS(size_x, size_y, static_obstacles=static_obss, used_dist= used_dist)
    all_path_lst = problem.run_loop(ag_start_pos_lst, task_lst)
    
    # --------------------------
    # Save animation
    # --------------------------

    if plot:
        animation_name = f'{size_x}x{size_y}_{len(static_obss)}obs_{len(ag_start_pos_lst)}ags_{len(task_lst)}tgts_{len(all_path_lst)}steps.gif'
        problem.plot_animation(animation_name, all_path_lst, small_env=True)
    
    return all_path_lst


def eval_env3(plot:bool = False, used_dist:str = 'manh',seed = 168, density_percent: int = 90):                
    # --------------------------
    # problem definition
    # --------------------------
    random.seed(seed)
    size_x = 35
    size_y = 21
    
    static_obss = []

    task_lst = [(0, 0), 
                (size_x-1, 0),
                (size_x-1, size_y-1), 
                (0, size_y-1),

                (int(size_x/3), 0), 
                (size_x-1, int(size_y/3)),
                (int(size_x*2/3), size_y-1),
                (0, int(size_y*2/3)-1),

                (int(size_x*2/3), 0),
                (size_x-1, int(size_y*2/3)-1),
                (int(size_x*1/3), size_y-1), 
                (0, int(size_y/3))]    
    
    tgt_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+task_lst))
    tgt_ag_start_pos_lst = random.sample(tgt_ag_option_nodes, len(task_lst))

    other_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+tgt_ag_start_pos_lst))
    max_available_nodes = len(other_ag_option_nodes)
    num_other_agents = _compute_other_agents(
        size_x,
        size_y,
        density_percent,
        density_boost=5,
        max_available=max_available_nodes,
    )
    other_ag_start_pos_lst = random.sample(other_ag_option_nodes, num_other_agents)

    ag_start_pos_lst = tgt_ag_start_pos_lst + other_ag_start_pos_lst

    # --------------------------
    # Run
    # --------------------------
    problem = PHANS(size_x, size_y, static_obstacles=static_obss, used_dist=used_dist)
    all_path_lst = problem.run_loop(ag_start_pos_lst, task_lst)
    
    # --------------------------
    # Save animation
    # --------------------------
    if plot:
        animation_name = f'{size_x}x{size_y}_{len(static_obss)}obs_{len(ag_start_pos_lst)}ags_{len(task_lst)}tgts_{len(all_path_lst)}steps.gif'
        problem.plot_animation(animation_name, all_path_lst, small_env=True)

    return all_path_lst

def eval_env4(plot:bool = False, used_dist:str = 'manh', seed = 168, density_percent: int = 90):                
    # --------------------------
    # problem definition
    # --------------------------
    random.seed(seed)
    size_x = 35
    size_y = 21

    static_obss = [(0, 9), (1, 9), (0, 10), (1, 10), (0, 11), (1, 11),
                   (4,  3), (5,  3), (6,  3), (4,  4), (5,  4), (6,  4), 
                   (4,  5), (5,  5), (6,  5), 
                   (4,  15), (5,  15), (6,  15), (4,  16), (5,  16), (6,  16), 
                   (4,  17), (5,  17), (6,  17), 
                   (9, 9), (10, 9), (11, 9), (12, 9), (9, 10), (10, 10), 
                   (11, 10), (12, 10), (9, 11), (10, 11), (11, 11), (12, 11), 
                   (16, 3), (17, 3), (18, 3), (16, 4), (17, 4), (18, 4), 
                   (16, 5), (17, 5), (18, 5), 
                   (16, 15), (17, 15), (18, 15), (16, 16), (17, 16), (18, 16), 
                   (16, 17), (17, 17), (18, 17),
                   (22, 9), (23, 9), (24, 9), (25, 9), (22, 10), (23, 10), 
                   (24, 10), (25, 10), (22, 11), (23, 11), (24, 11), (25, 11),
                   (28, 3), (29, 3), (30, 3), (28, 4), (29, 4), (30, 4), 
                   (28, 5), (29, 5), (30, 5), 
                   (28, 15), (29, 15), (30, 15), (28, 16), (29, 16), (30, 16), 
                   (28, 17), (29, 17), (30, 17),
                   (33, 9), (34, 9), (33, 10), (34, 10), (33, 11), (34, 11)]

    task_lst = [(0, 0), 
                (size_x-1, 0),
                (size_x-1, size_y-1), 
                (0, size_y-1),

                (int(size_x/3), 0), 
                (size_x-1, int(size_y/3)),
                (int(size_x*2/3), size_y-1),
                (0, int(size_y*2/3)-1),

                (int(size_x*2/3), 0),
                (size_x-1, int(size_y*2/3)-1),
                (int(size_x*1/3), size_y-1), 
                (0, int(size_y/3))]    
    
    tgt_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+task_lst))
    tgt_ag_start_pos_lst = random.sample(tgt_ag_option_nodes, len(task_lst))

    other_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+tgt_ag_start_pos_lst))
    max_available_nodes = len(other_ag_option_nodes)
    num_other_agents = _compute_other_agents(
        size_x,
        size_y,
        density_percent,
        static_obstacles_count=len(static_obss),
        density_boost=5,
        max_available=max_available_nodes,
    )
    other_ag_start_pos_lst = random.sample(other_ag_option_nodes, num_other_agents)

    ag_start_pos_lst = tgt_ag_start_pos_lst + other_ag_start_pos_lst

    # --------------------------
    # Run
    # --------------------------
    problem = PHANS(size_x, size_y, static_obstacles=static_obss, used_dist=used_dist)
    all_path_lst = problem.run_loop(ag_start_pos_lst, task_lst)
    
    # --------------------------
    # Save animation
    # --------------------------
    if plot:
        animation_name = f'{size_x}x{size_y}_{len(static_obss)}obs_{len(ag_start_pos_lst)}ags_{len(task_lst)}tgts_{len(all_path_lst)}steps.gif'
        problem.plot_animation(animation_name, all_path_lst, small_env=True)

    return all_path_lst