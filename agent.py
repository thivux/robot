import numpy as np

class SimpleAgent:

    def __init__(self, row, col, timestep, epsilon, max_range):
        self.row = row
        self.col = col
        self.timestep = timestep
        self.epsilon = epsilon
        self.max_range = max_range

    def get_action(self, s):
        return 1 + np.random.randint(4) # random number in [1, 4]

    def get_graph(self, roll_outs):
        # Return a uniform based on the roll_outs:
        list_of_nodes = []
        for i in range(len(roll_outs)):
            roll_out = roll_outs[i][0]
            success = roll_outs[i][1]
            for j in range(len(roll_out)):
                observation = (roll_out[j][0], roll_out[j][1], roll_out[j][2], roll_out[j][3])
                list_of_nodes += [observation]

        list_of_nodes = list(set(list_of_nodes))
        n_nodes = len(list_of_nodes)

        output = []
        for i in range(len(list_of_nodes)):
            for j in range(len(list_of_nodes)):
                output += [(list_of_nodes[i], list_of_nodes[j], 1.0/n_nodes)]

        return output
