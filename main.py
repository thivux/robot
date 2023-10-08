from gridworld_env import GridworldEnv
from agent import SimpleAgent
from extract_graph import build_graph
import os
import numpy as np
import math


def compare_graph(target_graph, output_graph):
    target_d = dict([((e[0], e[1]), e[2]) for e in target_graph]) #TODO: 3 phan tu trong e tuong ung voi gi?
    output_d = dict([((e[0], e[1]), e[2]) for e in output_graph])

    errors = []
    for u in target_d:
        predict = output_d.get(u, 0)
        errors += [target_d[u] - predict]

    for u in output_d:
        if u not in target_d:
            errors += [-output_d[u]]

    errors = np.array(errors)
    rmse = math.sqrt(np.mean(errors*errors))

    return rmse


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plan = 1
    row = 5
    col = 5
    epsilon = 0.2
    T = 10 #TODO: timestep?
    max_range = 2 

    env = GridworldEnv(plan=1, epislon=0.2, max_range=max_range)
    agent = SimpleAgent(row, col, T, epsilon, max_range=max_range)
    history = []
    for episode in range(10):
        s = env.reset() # list of 6 values : [1.0, 0.0, 1.0, 0.0, -0.5, 0.0]
        roll_out = [s] 
        success = False #TODO: get goal or not?? 

        for t in range(T):
            a = agent.get_action(s) # calculate the optimal action at state s
            s, r, done, info = env.step(a) # env's responses to the action 
            roll_out += [s]
            if done:
                success = True
                break

        history += [(roll_out, success)]

    # print(history)
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    input_file = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
    target_graph = build_graph(input_file, max_range, epsilon)
    output_graph = agent.get_graph(history)

    print("Errors: ", compare_graph(target_graph, output_graph))

    # TODO: Improve the errors
