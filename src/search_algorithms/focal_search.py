import heapq
from itertools import count
from math import fabs, sqrt
import numpy as np
from src.utils import Location, State


class FocalSearchFollowingConflict():
    def __init__(
        self,
        dimension: tuple[int, int],
        agent: dict,
        obstacles: set,
        moving_obstacles: list[tuple[int, int, int]] = None,
        moving_obstacle_edges: list[tuple[int, int, int, int]] = None,
        focal_max_iter: int = -1,
        agent_start_pos_lst: list[tuple[int, int]] = None,
        null_agent_pos_lst: list[tuple[int, int]] = None,
        is_dst_add: bool = True,
        considering_cycle_conflict: bool = True,
        used_dist: str = 'euclid',
        w: float = 1.5,  # Weight for suboptimality bound
    ):
        if moving_obstacles is None:
            moving_obstacles = []
        if moving_obstacle_edges is None:
            moving_obstacle_edges = []
        if agent_start_pos_lst is None:
            agent_start_pos_lst = []
        if null_agent_pos_lst is None:
            null_agent_pos_lst = []

        self.dimension = dimension
        self.obstacles = obstacles
        self.moving_obstacles = moving_obstacles
        self.moving_obstacle_edges = moving_obstacle_edges
        self.focal_max_iter = focal_max_iter
        self.agent_start_pos_lst = agent_start_pos_lst
        self.null_agent_pos_lst = null_agent_pos_lst
        self.is_dst_add = is_dst_add
        self.considering_cycle_conflict = considering_cycle_conflict
        self.used_dist = used_dist
        self.w = w  # Suboptimality bound

        start_state = State(0, Location(agent["start"][0], agent["start"][1]))
        goal_state = State(0, Location(agent["goal"][0], agent["goal"][1]))
        self.agent = {"start": start_state, "goal": goal_state}
        
        self.iter = 0

    def _get_neighbors(self, state: State) -> list[State]:
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y + 1))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y - 1))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x - 1, state.location.y))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x + 1, state.location.y))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        return neighbors

    def _get_all_obstacles(self, time: int) -> set[tuple[int, int]]:
        all_obs = set()
        for o in self.moving_obstacles:
            if o[2] < 0 and time >= -o[2]:
                all_obs.add((o[0], o[1]))
        return self.obstacles | all_obs

    def _state_valid(self, state: State) -> bool:
        return (
            state.location.x >= 0
            and state.location.x < self.dimension[0]
            and state.location.y >= 0
            and state.location.y < self.dimension[1]
            and (state.location.x, state.location.y)
            not in self._get_all_obstacles(state.time)
        )

    def _find_cycles(self, idx_vectors: list[tuple[int, int]]) -> bool:
        edge_map = {start: end for start, end in idx_vectors}

        for start in edge_map:
            visited = {start}
            current = start
            while True:
                if edge_map[current] in visited:
                    return True
                if edge_map[current] not in edge_map:
                    break
                visited.add(edge_map[current])
                current = edge_map[current]

        return False

    def _transition_valid(self, state_cur: State, state_next: State) -> bool:
        if (state_next.location.x, state_next.location.y, state_next.time) in (
            self.moving_obstacles
        ):
            return False
        if (state_next.location.x, state_next.location.y, state_next.time - 1) in (
            self.moving_obstacles
        ):
            return False
        if (state_next.location.x, state_next.location.y, state_next.time + 1) in (
            self.moving_obstacles
        ):
            return False
        if (
            state_next.location.x,
            state_next.location.y,
            state_cur.location.x,
            state_cur.location.y,
        ) in self.moving_obstacle_edges:
            return False
        # Cycle conflicts against moving obstacle (without considering time)
        if self.considering_cycle_conflict and len(self.moving_obstacle_edges) > 0:
            cur_pos_lst = [(state_cur.location.x, state_cur.location.y)] + [
                (mo[0], mo[1]) for mo in self.moving_obstacle_edges
            ]
            next_pos_lst = [(state_next.location.x, state_next.location.y)] + [
                (mo[2], mo[3]) for mo in self.moving_obstacle_edges
            ]

            dup_idx_pairs = []
            for i_cur_pos, tgt_ag_cur_pos in enumerate(cur_pos_lst):
                for i_next_pos, tgt_ag_next_pos in enumerate(next_pos_lst):
                    if i_cur_pos != i_next_pos and tgt_ag_cur_pos == tgt_ag_next_pos:
                        dup_idx_pairs.append((i_cur_pos, i_next_pos))

            if self._find_cycles(dup_idx_pairs):
                return False

        return True

    def __manhattan_dist(self, state:State, goal):
        return fabs(state.location.x - goal.location.x) + fabs(
            state.location.y - goal.location.y
        )

    def __euclid_dist(self, state:State, goal):
        return sqrt((state.location.x - goal.location.x)**2 + (state.location.y - goal.location.y)**2)
    
    def __chebyshev_dist(self, state:State, goal):
        return max(fabs(state.location.x - goal.location.x), fabs(state.location.y - goal.location.y))
    
    def __octile_dist(self, state:State, goal):
        dx = abs(state.location.x - goal.location.x)
        dy = abs(state.location.y - goal.location.y)
        return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
    
    def __mixed_max(self, state:State, goal):
        return max(self.__manhattan_dist(state, goal), self.__chebyshev_dist(state, goal))
    
    def __weighted_mixed(self, state:State, goal, alpha=0.5):
        return alpha * self.__manhattan_dist(state, goal) + (1 - alpha) * self.__chebyshev_dist(state, goal)
    
    def _admissible_heuristic(self, state: State) -> float:
        goal = self.agent["goal"]
        if self.used_dist == 'euclid':
            return self.__euclid_dist(state, goal)            
        elif self.used_dist == 'cheb':
            return self.__chebyshev_dist(state, goal)
        elif self.used_dist == 'octile':
            return self.__octile_dist(state, goal)
        elif self.used_dist == 'mixed':
            return self.__mixed_max(state, goal)
        elif self.used_dist == 'weighted':
            return self.__weighted_mixed(state, goal)
        else:
            return self.__manhattan_dist(state, goal)

    def _is_at_goal(self, state: State) -> bool:
        goal_state = self.agent["goal"]
        return state.is_equal_except_time(goal_state)

    def _reconstruct_path(self, came_from: dict, current: State) -> list[State]:
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def _secondary_heuristic(self, state: State) -> float:
        goal = self.agent["goal"]
        
        primary_h = self._admissible_heuristic(state)
        
        time_penalty = state.time * 0.01
        
        obstacle_penalty = 0
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = state.location.x + dx, state.location.y + dy
            if (nx, ny) in self.obstacles:
                obstacle_penalty += 0.05
        
        return primary_h + time_penalty + obstacle_penalty

    def _search(self) -> list[State]:
        initial_state = self.agent["start"]
        step_cost = 1

        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        g_score = {initial_state: 0}

        h_score = self._admissible_heuristic(initial_state)
        f_score = {initial_state: h_score}

        open_heap = []
        index = count(0)
        heapq.heappush(
            open_heap, (f_score[initial_state], h_score, next(index), initial_state)
        )
        
        focal_set = []
        min_f_score = f_score[initial_state]
        heapq.heappush(
            focal_set, (self._secondary_heuristic(initial_state), next(index), initial_state)
        )

        in_focal = {initial_state}

        while open_set and (self.focal_max_iter == -1 or self.iter < self.focal_max_iter):
            self.iter = self.iter + 1

            if open_heap:
                current_min_f = open_heap[0][0]
                
                if current_min_f > min_f_score * 1.2:  
                    min_f_score = current_min_f
                    focal_set = []
                    in_focal.clear()
                    index_focal = count(0)
                    
                    for (f_val, h_val, idx, node) in open_heap:
                        if f_val <= self.w * min_f_score:
                            heapq.heappush(
                                focal_set, 
                                (self._secondary_heuristic(node), next(index_focal), node)
                            )
                            in_focal.add(node)

            current = None
            if focal_set:
                while focal_set:
                    secondary_h, idx, node = heapq.heappop(focal_set)
                    if node in open_set and node in in_focal:
                        current = node
                        break
                
                if current is None:
                    focal_set = []
                    in_focal.clear()

            if current is None and open_heap:
                current = heapq.heappop(open_heap)[3]

            if current is None:
                break

            open_set.discard(current)
            in_focal.discard(current)
            


            if self._is_at_goal(current):
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            neighbor_list = self._get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                dst_add = 0
                if self.is_dst_add:
                    nx, ny = neighbor.location.x, neighbor.location.y
                    neighbor_xy = np.array((nx, ny))
                    null_xy = np.array(self.null_agent_pos_lst)
                    if len(null_xy) > 0:
                        dst_to_null = np.abs(null_xy - neighbor_xy).sum(axis=1).min()
                        if dst_to_null > 0 and dst_to_null > tentative_g_score:
                            dst_add = dst_to_null - tentative_g_score

                if neighbor not in open_set:
                    # Новый узел
                    open_set.add(neighbor)
                    came_from[neighbor] = current 
                    g_score[neighbor] = tentative_g_score
                    h_score = self._admissible_heuristic(neighbor)
                    new_f_score = tentative_g_score + h_score + dst_add
                    f_score[neighbor] = new_f_score
                    
                    # Добавляем в OPEN
                    heapq.heappush(
                        open_heap, 
                        (new_f_score, h_score, next(index), neighbor)
                    )
                    
                    # Добавляем в FOCAL если удовлетворяет условию
                    if new_f_score <= self.w * min_f_score:
                        heapq.heappush(
                            focal_set,
                            (self._secondary_heuristic(neighbor), next(index), neighbor)
                        )
                        in_focal.add(neighbor)
                        
                elif tentative_g_score < g_score.setdefault(neighbor, float("inf")):
                    # Обновление существующего узла
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_score = self._admissible_heuristic(neighbor)
                    new_f_score = tentative_g_score + h_score + dst_add
                    
                    # Обновляем OPEN - просто добавляем новую запись (старая будет проигнорирована при pop)
                    heapq.heappush(
                        open_heap, 
                        (new_f_score, h_score, next(index), neighbor)
                    )
                    
                    # Обновляем FOCAL
                    if neighbor in in_focal:
                        # Помечаем для удаления из FOCAL
                        in_focal.discard(neighbor)
                    
                    if new_f_score <= self.w * min_f_score:
                        heapq.heappush(
                            focal_set,
                            (self._secondary_heuristic(neighbor), next(index), neighbor)
                        )
                        in_focal.add(neighbor)
                        
                    f_score[neighbor] = new_f_score

        return []

    def compute_solution(self) -> list[dict]:
        local_solution = self._search()
        if not local_solution:
            return {}
        
        path_dict_list = [
            {"t": state.time, "x": state.location.x, "y": state.location.y}
            for state in local_solution
        ]
        
        return path_dict_list