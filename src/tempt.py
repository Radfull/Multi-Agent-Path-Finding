def run_loop(
        self,
        ag_start_pos_lst: list[tuple[int, int]],
        task_lst: list[tuple[int, int]],
        a_star_max_iter: int = -1,
        max_loop: int = 300
) -> list[list[tuple[int, int]]]:
    """
    Move agents step-by-step following the target paths.
    """
    # -----------------------------------
    # 1. Plan target paths
    # -----------------------------------
    self._init_loop(ag_start_pos_lst, task_lst, a_star_max_iter)

    # -----------------------------------
    # 2. Evacuate obstructing agents and move target agents
    # -----------------------------------

    # Initialize
    all_path_lst = [copy.deepcopy(self.ag_cur_pos_lst)]
    self.tgt_cur_steps = [0] * len(self.tgt_ag_path_lst)

    loop = 0
    self._update_goal_ag()
    tgt_ags_cur_pos_lst = self.ag_cur_pos_lst[: self.task_len]

    for i_tgt_ag in [i for i in range(self.task_len) if i not in self.goal_ag_idx_lst]:
        self.tgt_cur_steps[i_tgt_ag] = (
                len(self.tgt_ag_path_lst[i_tgt_ag])
                - self.tgt_ag_path_lst[i_tgt_ag][::-1].index(tgt_ags_cur_pos_lst[i_tgt_ag])
                - 1
        )

    # Start loop
    while (
            any(
                self.tgt_cur_steps[i] < len(self.tgt_ag_path_lst[i]) - 1
                for i in range(self.task_len)
            )
            and loop < max_loop
    ):
        # 🔥 ЗАМЕНИТЕ весь этот блок (строки 78-113):
        # Identify obstructing agents
        blocking_ag_pos_lst = []
        next_blocking_ag_dist_from_goal_lst = []

        for i_tgt_ag in range(self.task_len):
            if self.tgt_cur_steps[i_tgt_ag] < len(self.tgt_ag_path_lst[i_tgt_ag]) - 1:
                next_tgt_pos_lst = self.tgt_ag_path_lst[i_tgt_ag][
                                   self.tgt_cur_steps[i_tgt_ag] + 1:
                                   ]

                # Get the next blocking agent position
                for pos in next_tgt_pos_lst:
                    if pos in self.ag_cur_pos_lst and pos not in tgt_ags_cur_pos_lst:
                        dst_to_goal = (
                                len(self.tgt_ag_path_lst[i_tgt_ag])
                                - self.tgt_ag_path_lst[i_tgt_ag].index(pos)
                                + 1
                        )
                        if pos not in blocking_ag_pos_lst:
                            blocking_ag_pos_lst.append(pos)
                            next_blocking_ag_dist_from_goal_lst.append(dst_to_goal)
                        else:
                            idx = blocking_ag_pos_lst.index(pos)
                            if next_blocking_ag_dist_from_goal_lst[idx] < dst_to_goal:
                                next_blocking_ag_dist_from_goal_lst[idx] = dst_to_goal

        # Order the obstructing agents
        blocking_ag_pos_lst = [
            pos
            for _, pos in sorted(
                zip(
                    next_blocking_ag_dist_from_goal_lst,
                    blocking_ag_pos_lst,
                ),
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        # 🔥 НА ЭТО (одна строка):
        blocking_ag_pos_lst = self._find_blocking_agents_optimized()

        # Define dependency among target agents...
        # (остальной код оставьте без изменений)
        high_priority_blocking_ag_pos_lst = []
        dependency_relation = []  # (A, B): A depends on B

        for i_tgt_ag in [i for i in range(self.task_len) if i not in self.goal_ag_idx_lst]:
            tgt_ag_next_pos = self.tgt_ag_path_lst[i_tgt_ag][self.tgt_cur_steps[i_tgt_ag] + 1]
            tgt_ag_cur_idx = self.tgt_cur_steps[i_tgt_ag]

            for j_tgt_ag in range(self.task_len):
                if i_tgt_ag != j_tgt_ag:
                    from_idx = self.tgt_cur_steps[j_tgt_ag]
                    to_idx = min(len(self.tgt_ag_path_lst[j_tgt_ag]) - 1, tgt_ag_cur_idx)
                    if tgt_ag_next_pos in self.tgt_ag_path_lst[j_tgt_ag][from_idx:to_idx]:
                        if j_tgt_ag in [x for x, _ in dependency_relation]:
                            dependency_relation.append((i_tgt_ag, j_tgt_ag))
                        else:
                            dependency_relation = [(i_tgt_ag, j_tgt_ag)] + dependency_relation

        for i_tgt_ag, j_tgt_ag in dependency_relation:
            tgt_ag_next_pos = self.tgt_ag_path_lst[i_tgt_ag][self.tgt_cur_steps[i_tgt_ag] + 1]
            from_idx = self.tgt_cur_steps[j_tgt_ag]
            high_priority_blocking_ag_pos_lst += self.tgt_ag_path_lst[j_tgt_ag][
                                                 from_idx: min(
                                                     len(self.tgt_ag_path_lst[j_tgt_ag]),
                                                     self.tgt_ag_path_lst[j_tgt_ag].index(tgt_ag_next_pos) + 2,
                                                 )
                                                 ][::-1]

        for pos in high_priority_blocking_ag_pos_lst:
            if pos in blocking_ag_pos_lst:
                blocking_ag_pos_lst.remove(pos)
                blocking_ag_pos_lst = [pos] + blocking_ag_pos_lst

        # Choose the empty vertices
        # 🔥 ЗАМЕНИТЕ эту строку:
        self._update_null_ag_pos_lst()
        # 🔥 НА ЭТУ:
        self._update_null_ag_pos_lst_optimized()

        available_null_ags_pos_lst = copy.deepcopy(list(set(self.null_ag_pos_lst)))
        selected_null_ags_pos_lst = []

        for i, bag_pos in enumerate(blocking_ag_pos_lst):
            # Generally choose the null agent with the smallest h-value,
            # but avoid interfering with target agents' movements.

            dst_from_bag_to_goal = float("inf")
            for i_tgt_ag in range(self.task_len):
                if bag_pos in self.tgt_ag_path_lst[i_tgt_ag]:
                    dst_from_bag_to_goal = min(
                        dst_from_bag_to_goal,
                        len(self.tgt_ag_path_lst[i_tgt_ag])
                        - self.tgt_ag_path_lst[i_tgt_ag].index(bag_pos)
                        - 1,
                    )

            # Preserve vertices between the target and obstructing agents on target paths
            tgt_preserved_pos = []
            dst_penalty_dct = {}

            for i_tgt_ag in range(self.task_len):
                tgt_idx_on_tgt_ag_path = self.tgt_cur_steps[i_tgt_ag]

                if bag_pos in self.tgt_ag_path_lst[i_tgt_ag]:
                    bag_idx_on_tgt_ag_path = self.tgt_ag_path_lst[i_tgt_ag].index(bag_pos)
                    for pos in self.tgt_ag_path_lst[i_tgt_ag][bag_idx_on_tgt_ag_path:]:
                        dst_from_null_to_tgt = np.sum(
                            np.abs(np.array(pos)
                                   - np.array(self.tgt_ag_path_lst[i_tgt_ag] \
                                                  [self.tgt_cur_steps[i_tgt_ag]])
                                   )
                        )
                        if pos in dst_penalty_dct.keys():
                            dst_penalty_dct[pos] = max(
                                dst_penalty_dct[pos],
                                dst_from_null_to_tgt
                            )
                        else:
                            dst_penalty_dct[pos] = dst_from_null_to_tgt

                # For null agents from the target's current position to the goal:
                for pos in self.tgt_ag_path_lst[i_tgt_ag][tgt_idx_on_tgt_ag_path:]:
                    dst_to_goal = (
                            len(self.tgt_ag_path_lst[i_tgt_ag])
                            - self.tgt_ag_path_lst[i_tgt_ag].index(pos)
                            - 1
                    )
                    if dst_from_bag_to_goal < dst_to_goal:
                        tgt_preserved_pos.append(pos)

            tmp_available_null_ags_pos_lst = [
                null_ag_pos
                for null_ag_pos in available_null_ags_pos_lst
                if null_ag_pos not in tgt_preserved_pos
            ]

            if len(tmp_available_null_ags_pos_lst) > 0:
                # 🔥 ЗАМЕНИТЕ эти строки:
                dst_penalty_array = np.array(
                    [dst_penalty_dct.get(pos, 0) for pos in tmp_available_null_ags_pos_lst]
                )
                distances = (
                                np.abs(np.array(tmp_available_null_ags_pos_lst) - np.array(bag_pos))
                                .sum(axis=1)
                            ) + dst_penalty_array
                min_index = np.argmin(distances)
                pos = tmp_available_null_ags_pos_lst[min_index]
                # 🔥 НА ЭТИ:
                min_index, closest_pos = self._get_closest_null_ag_pos_optimized(bag_pos)
                if min_index != -1 and closest_pos in tmp_available_null_ags_pos_lst:
                    pos = closest_pos
                    selected_null_ags_pos_lst.append(pos)
                    available_null_ags_pos_lst.remove(pos)
                else:
                    break
            else:
                break

        # 🔥 ОСТАЛЬНОЙ КОД ОСТАВЬТЕ БЕЗ ИЗМЕНЕНИЙ
        # Plan paths for null agents to move to obstructing agents' positions.
        dimension = (self.size_x, self.size_y)
        obss = self.static_obss + list(
            set(tgt_ags_cur_pos_lst)
            - set(selected_null_ags_pos_lst)
            - set(blocking_ag_pos_lst)
        )
        ags = []
        for i in range(len(selected_null_ags_pos_lst)):
            if len(ags) == 0 or blocking_ag_pos_lst[i] not in [ag["goal"] for ag in ags]:
                ags.append(
                    {
                        "start": selected_null_ags_pos_lst[i],
                        "goal": blocking_ag_pos_lst[i],
                        "name": i,
                    }
                )

        mov_obss = []
        solution_dict_all = []
        for ag in ags:
            tgt_preserved_pos = []
            for i_tgt_ag in range(self.task_len):
                if ag["goal"] in self.tgt_ag_path_lst[i_tgt_ag]:
                    tgt_preserved_pos += [
                        pos
                        for pos in self.tgt_ag_path_lst[i_tgt_ag][
                                   self.tgt_cur_steps[i_tgt_ag]: self.tgt_ag_path_lst[
                                       i_tgt_ag
                                   ].index(ag["goal"])
                                   ]
                    ]

            env = self._create_search_algorithm(
                ag=ag,
                obstacles=set(obss + tgt_preserved_pos),
                moving_obstacles=mov_obss,
                moving_obstacle_edges=[],
                a_star_max_iter=10000
            )
            env.is_dst_add = False
            solution = env.compute_solution()
            mov_obss += [(s["x"], s["y"], s["t"]) for s in solution]
            solution_dict_all.append(solution)

        null_ags_path_lst = []
        for solution_dict in solution_dict_all:
            null_ags_path_lst.append([(s['x'], s['y']) for s in solution_dict])

        # Execute movements
        ag_next_pos_lst = copy.deepcopy(self.ag_cur_pos_lst)
        ag_pre_pos_lst = []

        one_tgt_ag_goal_flag = True
        while one_tgt_ag_goal_flag and loop < max_loop:
            # Move target agents if possible
            for i_tgt_ag in [
                i for i in range(self.task_len) if i not in self.goal_ag_idx_lst
            ]:
                tgt_ag_next_pos = self.tgt_ag_path_lst[i_tgt_ag][
                    self.tgt_cur_steps[i_tgt_ag] + 1
                    ]
                if (
                        tgt_ag_next_pos not in ag_next_pos_lst
                        and tgt_ag_next_pos not in ag_pre_pos_lst
                ):
                    ag_pre_pos_lst.append(copy.deepcopy(ag_next_pos_lst[i_tgt_ag]))
                    ag_next_pos_lst[i_tgt_ag] = copy.deepcopy(tgt_ag_next_pos)
                    one_tgt_ag_goal_flag = False

            # Move null agents by one step if possible
            for i_null_ag in range(len(null_ags_path_lst)):
                if len(null_ags_path_lst[i_null_ag]) > 1:
                    null_ag_pos_idx_lst_on_null_ag_path = [
                        i
                        for i, pos in enumerate(null_ags_path_lst[i_null_ag])
                        if pos not in self.ag_cur_pos_lst[self.task_len:]
                           and pos not in ag_next_pos_lst[self.task_len:]
                    ]

                    if len(null_ag_pos_idx_lst_on_null_ag_path) > 0:
                        null_ag_cur_pos_idx = max(null_ag_pos_idx_lst_on_null_ag_path)
                        null_ag_cur_pos = null_ags_path_lst[i_null_ag][null_ag_cur_pos_idx]

                        if (
                                null_ag_cur_pos not in ag_next_pos_lst
                                and null_ag_cur_pos_idx < len(null_ags_path_lst[i_null_ag]) - 1
                        ):
                            null_ag_next_pos = null_ags_path_lst[i_null_ag][
                                null_ag_cur_pos_idx + 1
                                ]

                            if (
                                    null_ag_next_pos in self.ag_cur_pos_lst
                                    and null_ag_next_pos in ag_next_pos_lst
                            ):
                                ag_next_pos_lst[
                                    ag_next_pos_lst.index(null_ag_next_pos)
                                ] = copy.deepcopy(null_ag_cur_pos)
                                null_ags_path_lst[i_null_ag] = null_ags_path_lst[i_null_ag][1:]

            self.tgt_ag_path_lst_plot.append(copy.deepcopy(self.tgt_ag_path_lst))
            all_path_lst.append(copy.deepcopy(ag_next_pos_lst))
            self.ag_cur_pos_lst = copy.deepcopy(ag_next_pos_lst)

            loop += 1

        self._update_goal_ag()
        tgt_ags_cur_pos_lst = self.ag_cur_pos_lst[: self.task_len]
        for i_tgt_ag in [i for i in range(self.task_len) if i not in self.goal_ag_idx_lst]:
            self.tgt_cur_steps[i_tgt_ag] = (
                    len(self.tgt_ag_path_lst[i_tgt_ag])
                    - self.tgt_ag_path_lst[i_tgt_ag][::-1].index(
                tgt_ags_cur_pos_lst[i_tgt_ag]
            )
                    - 1
            )
        self._update_goal_ag()

    return all_path_lst