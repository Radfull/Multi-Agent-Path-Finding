import heapq
from itertools import count
from math import fabs, sqrt

from src.utils import Location, State


class BiAStarFollowingConflict:

    def __init__(
        self,
        dimension: tuple[int, int],
        agent: dict,
        obstacles: set,
        moving_obstacles=None,
        moving_obstacle_edges=None,
        a_star_max_iter: int = -1,
        used_dist: str = "manh",
        weight: float = 1.0,
        **_,
    ):
        if moving_obstacles is None:
            moving_obstacles = []
        if moving_obstacle_edges is None:
            moving_obstacle_edges = []

        self.dimension = dimension
        self.obstacles = obstacles
        self.moving_obstacles = moving_obstacles
        self.moving_obstacle_edges = moving_obstacle_edges
        self.a_star_max_iter = a_star_max_iter
        self.max_iter = a_star_max_iter
        self.used_dist = used_dist
        self.weight = min(1.0, weight if weight is not None else 1.0)

        self.start_state = State(0, Location(agent["start"][0], agent["start"][1]))
        self.goal_state = State(0, Location(agent["goal"][0], agent["goal"][1]))

        self.iter = 0
        self.counter = count(0)

    def _neighbors(self, loc: Location) -> list[Location]:
        candidates = [
            Location(loc.x + 1, loc.y),
            Location(loc.x - 1, loc.y),
            Location(loc.x, loc.y + 1),
            Location(loc.x, loc.y - 1),
        ]
        valid = []
        for n in candidates:
            if (
                0 <= n.x < self.dimension[0]
                and 0 <= n.y < self.dimension[1]
                and (n.x, n.y) not in self.obstacles
            ):
                valid.append(n)
        return valid

    def _heuristic(self, loc: Location, goal: Location) -> float:
        if self.used_dist == "euclid":
            return sqrt((loc.x - goal.x) ** 2 + (loc.y - goal.y) ** 2)
        if self.used_dist == "cheb":
            return max(fabs(loc.x - goal.x), fabs(loc.y - goal.y))
        if self.used_dist == "octile":
            dx = abs(loc.x - goal.x)
            dy = abs(loc.y - goal.y)
            return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
        if self.used_dist == "mixed":
            manh = fabs(loc.x - goal.x) + fabs(loc.y - goal.y)
            cheb = max(fabs(loc.x - goal.x), fabs(loc.y - goal.y))
            return max(manh, cheb)
        if self.used_dist == "weighted":
            manh = fabs(loc.x - goal.x) + fabs(loc.y - goal.y)
            cheb = max(fabs(loc.x - goal.x), fabs(loc.y - goal.y))
            return 0.5 * manh + 0.5 * cheb
        return fabs(loc.x - goal.x) + fabs(loc.y - goal.y)

    def _push(self, heap, f, g, loc, g_map):
        if (
            (loc.x, loc.y) not in g_map
            or g < g_map[(loc.x, loc.y)]
        ):
            g_map[(loc.x, loc.y)] = g
            heapq.heappush(heap, (f, next(self.counter), loc))
            return True
        return False

    def _reconstruct(self, meet_loc, came_from_f, came_from_b):
        forward_path = []
        cur_key = (meet_loc.x, meet_loc.y)
        cur = meet_loc
        while cur_key in came_from_f:
            forward_path.append(cur)
            cur = came_from_f[cur_key]
            cur_key = (cur.x, cur.y)
        forward_path.append(cur)  # start
        forward_path = forward_path[::-1]

        backward_path = []
        cur_key = (meet_loc.x, meet_loc.y)
        cur = meet_loc
        while cur_key in came_from_b:
            cur = came_from_b[cur_key]
            cur_key = (cur.x, cur.y)
            backward_path.append(cur)

        full_path = forward_path + backward_path
        result = []
        for t, loc in enumerate(full_path):
            result.append({"t": t, "x": loc.x, "y": loc.y})
        return result

    def compute_solution(self) -> list[dict]:
        start_loc = self.start_state.location
        goal_loc = self.goal_state.location

        forward_open = []
        backward_open = []

        forward_came = {}
        backward_came = {}

        forward_g = {(start_loc.x, start_loc.y): 0}
        backward_g = {(goal_loc.x, goal_loc.y): 0}

        forward_closed = set()
        backward_closed = set()

        heapq.heappush(
            forward_open,
            (
                self.weight * self._heuristic(start_loc, goal_loc),
                next(self.counter),
                start_loc,
            ),
        )
        heapq.heappush(
            backward_open,
            (
                self.weight * self._heuristic(goal_loc, start_loc),
                next(self.counter),
                goal_loc,
            ),
        )

        best_cost = float("inf")
        meeting_point = None
        expanded = 0
        last_meeting_iteration = -1

        while forward_open and backward_open:
            if self.a_star_max_iter != -1 and expanded >= self.a_star_max_iter:
                break

            f_fwd = forward_open[0][0]
            f_bwd = backward_open[0][0]

            if best_cost <= min(f_fwd, f_bwd):
                break

            if f_fwd <= f_bwd:
                _, _, current = heapq.heappop(forward_open)
                current_key = (current.x, current.y)
                if current_key in forward_closed:
                    continue
                forward_closed.add(current_key)

                if current_key in backward_closed:
                    total_cost = forward_g[current_key] + backward_g[current_key]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_point = current
                        last_meeting_iteration = expanded
                    continue

                for neighbor in self._neighbors(current):
                    neighbor_key = (neighbor.x, neighbor.y)
                    tentative_g = forward_g[current_key] + 1
                    f_cost = tentative_g + self.weight * self._heuristic(neighbor, goal_loc)
                    if self._push(forward_open, f_cost, tentative_g, neighbor, forward_g):
                        forward_came[(neighbor.x, neighbor.y)] = current

                    if neighbor_key in backward_g:
                        total_cost = tentative_g + backward_g[neighbor_key]
                        if total_cost < best_cost:
                            best_cost = total_cost
                            meeting_point = neighbor
                            last_meeting_iteration = expanded
            else:
                _, _, current = heapq.heappop(backward_open)
                current_key = (current.x, current.y)
                if current_key in backward_closed:
                    continue
                backward_closed.add(current_key)

                if current_key in forward_closed:
                    total_cost = backward_g[current_key] + forward_g[current_key]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        meeting_point = current
                        last_meeting_iteration = expanded
                    continue

                for neighbor in self._neighbors(current):
                    neighbor_key = (neighbor.x, neighbor.y)
                    tentative_g = backward_g[current_key] + 1
                    f_cost = tentative_g + self.weight * self._heuristic(neighbor, start_loc)
                    if self._push(backward_open, f_cost, tentative_g, neighbor, backward_g):
                        backward_came[(neighbor.x, neighbor.y)] = current

                    if neighbor_key in forward_g:
                        total_cost = tentative_g + forward_g[neighbor_key]
                        if total_cost < best_cost:
                            best_cost = total_cost
                            meeting_point = neighbor
                            last_meeting_iteration = expanded

            expanded += 1

        if meeting_point is None:
            return []

        return self._reconstruct(meeting_point, forward_came, backward_came)

