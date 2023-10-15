from gridworld_env import GridworldEnv
from agent import DQNAgent
from extract_graph import build_graph
import os
import numpy as np
import math
import wandb
import itertools


def compare_graph(target_graph, output_graph):
    target_d = dict([((e[0], e[1]), e[2]) for e in target_graph])
    output_d = dict([((e[0], e[1]), e[2]) for e in output_graph])

    errors = []
    for u in target_d:
        predict = output_d.get(u, 0)
        errors += [target_d[u] - predict]

    for u in output_d:
        if u not in target_d:
            errors += [-output_d[u]]

    errors = np.array(errors)
    rmse = math.sqrt(np.mean(errors * errors))

    return rmse


if __name__ == '__main__':
    plan_values = [1, 2]
    epsilon_values = [0.2, 0.3] 
    T_values = [4, 6, 8, 10]
    max_range_values = [2, 3, 4, 5] 
    threshold_values = [0.2, 0.5, 0.8]

    config_combinations = itertools.product(
        plan_values, epsilon_values, T_values, max_range_values, threshold_values
    )

    for config in config_combinations:
        plan, epsilon, T, max_range, threshold = config
        seed = 1
        print(f'config: {config}')

        env = GridworldEnv(plan=1, epislon=0.2, max_range=max_range, seed=seed)
        state_dim = len(env.reset())  # 6
        action_dim = env.action_space.n  # 5
        agent = DQNAgent(state_dim, action_dim, threshold=threshold)
        history = []
        episodes = 10000

        this_file_path = os.path.dirname(os.path.realpath(__file__))
        input_file = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
        target_graph = build_graph(input_file, max_range, epsilon)

        # Log configuration parameters to WandB
        config_dict = {
            "plan": plan,
            "epsilon": epsilon,
            "T": T,
            "max_range": max_range,
            "threshold": threshold,
            "seed": seed,
        }
        experiment_name = "_".join(
            [f"{key}={value}" for key, value in config_dict.items()])
        wandb.init(project='robot', entity='diogenes-student',
                   config=config_dict, name=experiment_name, 
                   tags=[f"plan_{plan}", f"epsilon_{epsilon}", f"T_{T}", f"max_range_{max_range}", f"threshold_{threshold}", f'seed_{seed}'])

        for e in range(episodes):
            s = env.reset()
            roll_out = [s]
            success = False

            for t in range(T):
                a = agent.get_action(s)
                s_next, r, done, info = env.step(a)
                agent.store_experience(s, a, r, s_next, done)
                s = s_next
                roll_out += [s]

                if done:
                    success = True
                    break

                # Train the agent after each step (you can also train after each episode)
                agent.train()

            history += [(roll_out, success)]

            # Update the target network periodically (e.g., every N episodes)
            if e % 500 == 0:
                output_graph = agent.get_graph(history)
                rmse = compare_graph(target_graph, output_graph)

                print("Errors: ", rmse)
                wandb.log({"RMSE": rmse})

                agent.update_target_network()

        wandb.finish()
        print('done an experiment')
