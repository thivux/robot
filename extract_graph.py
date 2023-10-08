from typing import List, Tuple
import numpy as np

TARGET = GREEN = 3
WALL = 1

action_pos_dict = {0: [-1, 0], 2: [1, 0], 1: [0, -1], 3: [0, 1]}
reverse_d = {0: 2, 2:0, 1:3, 3:1}

def read_grid_map(grid_map_path):
    # Return the gridmap imported from a txt plan

    grid_map = open(grid_map_path, 'r').readlines()
    grid_map_array = []
    for k1 in grid_map:
        k1s = k1.split(' ')
        tmp_arr = []
        for k2 in k1s:
            try:
                tmp_arr.append(int(k2))
            except:
                pass
        grid_map_array.append(tmp_arr)
    grid_map_array = np.array(grid_map_array, dtype=int)
    return grid_map_array


def generate_edges(grid_map, x, y, d, epsilon):
    output = []

    x1 = x + action_pos_dict[d][0]
    y1 = y + action_pos_dict[d][1]
    if grid_map[x1][y1] != WALL:
        output += [((x, y), (x1, y1), 1-2*epsilon)]

    d1 = (d - 1 + 4)%4
    x1 = x + action_pos_dict[d1][0]
    y1 = y + action_pos_dict[d1][1]
    if grid_map[x1][y1] != WALL:
        output += [((x, y), (x1, y1), epsilon)]
    else:
        output += [((x, y), (x, y), epsilon)]

    d1 = (d + 1)%4
    x1 = x + action_pos_dict[d1][0]
    y1 = y + action_pos_dict[d1][1]
    if grid_map[x1][y1] != WALL:
        output += [((x, y), (x1, y1), epsilon)]
    else:
        output += [((x, y), (x, y), epsilon)]

    return output

def build_fullObservable_graph(grid_map, epsilon) -> List[Tuple]:
    """
    With each [i, j] coodinates has grid_map[i, j] = 0, compute the optimal direction
    Then use it to extract the fully observable graph
    1 1 1 1 1 1
    1 3 0 0 1 1
    1 0 1 0 0 1
    1 0 1 0 1 1
    1 0 1 0 1 1
    1 1 1 1 1 1
    and epsilon = 10
    we would have:
    (2, 1) -> (1, 1): p = 0.8
    (2, 1) -> (2, 1): p = 0.2 (both left, and right stay at the same point
    :param grid_map:
    :param epsilon:
    :return:
    """
    # TODO: Use DFS or BFS to extract the fully version of the Markov Process Graph
    target_state = np.where(grid_map == TARGET)

    target_state = (target_state[0][0], target_state[1][0])
    visited = np.zeros_like(grid_map)
    visited[target_state[0]][target_state[1]] = 1

    # Run BFS to extract the graph
    queue = [target_state]
    list_edges = []
    while (len(queue) > 0):
        at = queue.pop()
        for k in range(4):
            dx = action_pos_dict[k][0]
            dy = action_pos_dict[k][1]

            x = at[0] + dx
            y = at[1] + dy
            if grid_map[x][y] == 0 and visited[x][y] == 0:
                visited[x][y] = 1
                queue.append((x, y))
                list_edges += generate_edges(grid_map, x, y, reverse_d[k], epsilon)

    return list_edges


def _fake_move(grid_map, at, action, d):
    x = at[0]
    y = at[1]
    for i in range(d):
        x += action_pos_dict[action][0]
        y += action_pos_dict[action][1]
        target_position = grid_map[x, y]
        if target_position == WALL:
            return i

    return d-1

def get_observation(grid_map, at, max_range):
    """

    :param grid_map:
    :param at:
    :param max_range:
    :return:
    """

    # TODO: Extract observation from a specific coordinate
    output = [0, 0, 0, 0]
    for k in range(0, 4):
        output[k] = _fake_move(grid_map, at, k, max_range)

    return tuple(output)


def build_graph(input_file: str, max_range: int, epsilon: float) -> List[Tuple]:
    """

    :param input_file:
    :param max_range:
    :param epsilon:
    :return:
        list of edges in form of:
        (start_node, end_node, transition_probablities)
    """

    # Step 1: Read the grid map
    grid_map = read_grid_map(input_file)

    # Step 2: Build the fully-observable version of the graph
    fully_graph = build_fullObservable_graph(grid_map, epsilon)

    # Step 3: Based on the fully-observable version of the graph, build the mdp graph
    list_edges = {}
    for edges in fully_graph:
        start_node = edges[0]
        end_node = edges[1]
        p = edges[2]

        partially_start_node = get_observation(grid_map, start_node, max_range)
        partially_end_node = get_observation(grid_map, end_node, max_range)

        # TODO: Update the list edges with (partially_starts_node, partially_end_node, p)
        if partially_start_node not in list_edges:
            list_edges[partially_start_node] = []

        list_edges[partially_start_node] += [(partially_end_node, p)]

    # TODO: Normalize the list_edges
    all_nodes = list_edges.keys()
    output = []
    for node in all_nodes:
        out_edges = list_edges[node]
        # TODO: Normalize the output probabilites to sum to 1.
        end_nodes = {}
        sum_p = 0
        for e in out_edges:
            if e[0] in end_nodes:
                end_nodes[e[0]] += e[1]
                sum_p += e[1]
            else:
                end_nodes[e[0]] = e[1]
                sum_p += e[1]
        #------------

        # Added to the output lists
        if sum_p > 0:
            for v in end_nodes:
                p = end_nodes[v] / sum_p
                output += [(node, v, p)]

    return output


if __name__=="__main__":
    output = build_graph("./plan1.txt", 2, 0.1)
    print(output)
