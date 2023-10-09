import wandb
import numpy as np
import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
from net import QNetwork


class SimpleAgent:

    def __init__(self, row, col, timestep, epsilon, max_range):
        self.row = row
        self.col = col
        self.timestep = timestep
        self.epsilon = epsilon
        self.max_range = max_range

    def get_action(self, state):
        can_go = [index for index, element in enumerate(
            state[:-2]) if element != 0]
        return 1 + random.choice(can_go)
        # return 1 + np.random.randint(4) # random number in [1, 4]

    def get_graph(self, roll_outs):
        # Return a uniform based on the roll_outs:
        list_of_nodes = []
        for i in range(len(roll_outs)):
            roll_out = roll_outs[i][0]
            success = roll_outs[i][1]
            for j in range(len(roll_out)):
                observation = (roll_out[j][0], roll_out[j]
                               [1], roll_out[j][2], roll_out[j][3])
                list_of_nodes += [observation]

        list_of_nodes = list(set(list_of_nodes))
        n_nodes = len(list_of_nodes)

        output = []
        for i in range(len(list_of_nodes)):
            for j in range(len(list_of_nodes)):
                output += [(list_of_nodes[i], list_of_nodes[j], 1.0/n_nodes)]

        return output


class DQNAgent:
    def __init__(self, state_dim, action_dim, threshold=1.0, threshold_decay=0.995, threshold_min=0.01, gamma=0.99, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.threshold = threshold  # probability the agent go random 
        self.threshold_decay = threshold_decay
        self.threshold_min = threshold_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=1000)  # Experience replay buffer
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        # Adjust the learning rate as needed
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.threshold:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(
                    torch.tensor(state, dtype=torch.float32))
                return np.argmax(q_values.numpy())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values for the current and next states
        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        # Compute the target Q-values using the Bellman equation
        targets = rewards + (1 - dones) * self.gamma * \
            torch.max(next_q_values, dim=1)[0]

        # Compute the loss (Huber loss)
        loss = torch.nn.functional.smooth_l1_loss(
            q_values.gather(1, actions.unsqueeze(1)), targets.unsqueeze(1))

        wandb.log({"Loss": loss.item()})

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_threshold(self):
        # Decay epsilon to reduce exploration over time
        if self.threshold > self.threshold_min:
            self.threshold *= self.threshold_decay

    def update_target_network(self):
        # Update the target network by copying the Q-network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.update_threshold()

    def get_graph(self, roll_outs):
        list_of_nodes = []
        for i in range(len(roll_outs)):
            roll_out = roll_outs[i][0]
            for j in range(len(roll_out)):
                observation = (roll_out[j][0], roll_out[j]
                               [1], roll_out[j][2], roll_out[j][3])
                list_of_nodes += [observation]
        list_of_nodes = list(set(list_of_nodes))

        adj_list = {}
        for i in range(len(list_of_nodes)):
            adj_list[list_of_nodes[i]] = []

        for i in range(len(roll_outs)):
            roll_out = roll_outs[i][0]
            success = roll_outs[i][1]
            for j in range(len(roll_out)):
                if j != len(roll_out) - 1:
                    observation1 = (
                        roll_out[j][0], roll_out[j][1], roll_out[j][2], roll_out[j][3])
                    observation2 = (
                        roll_out[j + 1][0], roll_out[j + 1][1], roll_out[j + 1][2], roll_out[j + 1][3])
                    adj_list[observation1].append(observation2)

        output = []
        for i in range(len(list_of_nodes)):
            if len(adj_list[list_of_nodes[i]]) == 0:
                for j in range(len(list_of_nodes)):
                    output += [(list_of_nodes[i], list_of_nodes[j], 0)]
            else:
                for j in range(len(list_of_nodes)):
                    cnt = adj_list[list_of_nodes[i]].count(list_of_nodes[j])
                    p = cnt / len(adj_list[list_of_nodes[i]])
                    output += [(list_of_nodes[i], list_of_nodes[j], p)]

        return output
