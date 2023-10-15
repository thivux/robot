import gym
import sys
import os
import copy
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding

EMPTY = BLACK = 0
WALL = GRAY = 1
TARGET = GREEN = 3
AGENT = RED = 0
SUCCESS = PINK = 6
COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5], GREEN: [0.0, 1.0, 0.0],
          RED: [1.0, 0.0, 0.0], PINK: [1.0, 0.0, 1.0]}

NOOP = 0
DOWN = 3
UP = 1
LEFT = 2
RIGHT = 4

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GridworldEnv:
    metadata = {'render.modes': ['human', 'rgb_array']}
    num_env = 0

    def __init__(self, plan, epislon, max_range=3, seed=1):

        self.epsilon = epislon # noise
        self.max_range = max_range # agent's max view
        self.rng = np.random.RandomState(seed) # randomize agent's start position

        self.actions = [NOOP, UP, LEFT, DOWN, RIGHT] 
        self.inv_actions = [0, 2, 1, 4, 3] # inversed actions
        self.action_space = spaces.Discrete(5) # action space is [0, 1, 2, 3, 4]
        self.action_pos_dict = {NOOP: [0, 0], UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1]} # [row, col]

        self.img_shape = [256, 256, 3]  # visualize state # ????

        # initialize system state
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
        self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), \
                                            high=np.array([1.0, 1.0, 1.0])) # why 3d space? 

        # agent state: start, target, current state
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state()
        self.agent_state = copy.deepcopy(self.agent_start_state)

        # set other parameters
        self.restart_once_done = False  # restart or not once done

        # set seed
        self.seed()

        # consider total episode reward
        self.episode_total_reward = 0.0

        # consider viewer for compatibility with gym
        self.viewer = None

    def seed(self, seed=None):

        # Fix seed for reproducibility

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _fake_move(self, action, d):
        x = self.agent_state[0]
        y = self.agent_state[1]
        for i in range(d):
            x += self.action_pos_dict[action][0]
            y += self.action_pos_dict[action][1]
            target_position = self.current_grid_map[x, y]
            if target_position == WALL:
                return i

        return d-1

    def get_state(self, coordinates, action, reward):

        '''
        return max step agent can go in up, down, left, right directions,
        the action taken by the agent (after -2.5 and / 5) and the reward
        '''

        ## Normalized for better perform of the NN
        fake_lidar = [0, 0, 0, 0, 0] # max step agent can go in 4 directions
        for i in range(1, 5):
            fake_lidar[i] = self._fake_move(i, self.max_range)

        return np.asarray([fake_lidar[1], fake_lidar[2], fake_lidar[3], fake_lidar[4],\
                           (action - 2.5) / 5., reward])

    def step(self, action):

        # Return next observation, reward, finished, success

        action = int(action) # ban dau la da int roi ma? 
        if action > 0:
            _e = self.rng.rand()
            if _e < self.epsilon:
                action -= 1
                if action == 0:
                    action = 4
            elif _e < 2*self.epsilon:
                action += 1
                if action > 4:
                    action = 1

        info = {'success': False}
        done = False

        # Penalties
        penalty_step = 0.01 # the cose of taking a step 
        penalty_wall = 0.05 # the cost of heading towards wall

        reward = -penalty_step
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                           self.agent_state[1] + self.action_pos_dict[action][1])

        if action == NOOP:
            info['success'] = True
            self.episode_total_reward += reward  # Update total reward
            return self.get_state(self.agent_state, action, reward), reward, False, info

        # Make a step
        next_state_out_of_map = (nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]) or \
                                (nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1])

        if next_state_out_of_map:
            info['success'] = False
            self.episode_total_reward += reward  # Update total reward
            return self.get_state(self.agent_state, action, reward), reward, False, info

        # successful behavior
        target_position = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]

        if target_position == EMPTY:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = AGENT

        elif target_position == WALL:

            info['success'] = False
            self.episode_total_reward += (reward - penalty_wall)  # Update total reward
            return self.get_state(self.agent_state, action, reward - penalty_wall), (reward - penalty_wall), False, info

        elif target_position == TARGET:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = SUCCESS

        self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
        self.agent_state = copy.deepcopy(nxt_agent_state)
        info['success'] = True

        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1]:
            done = True
            reward += 1.0
            if self.restart_once_done:
                self.reset()

        self.episode_total_reward += reward  # Update total reward
        return self.get_state(self.agent_state, action, reward), reward, done, info

    def reset(self):

        # Return the initial state of the environment

        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.episode_total_reward = 0.0
        return self.get_state(self.agent_state, 0.0, 0.0)

    def close(self):
        if self.viewer: self.viewer.close()

    def _read_grid_map(self, grid_map_path):

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

    def _get_agent_start_target_state(self):
        start_state = np.where(self.start_grid_map == AGENT)
        target_state = np.where(self.start_grid_map == TARGET)
        i = self.rng.randint(0, len(start_state[0]))

        start_state = (start_state[0][i], start_state[1][i])
        target_state = (target_state[0][0], target_state[1][0])

        return start_state, target_state

    def _gridmap_to_image(self, img_shape=None):

        # Return image from the gridmap

        if img_shape is None:
            img_shape = self.img_shape
        observation = np.random.randn(*img_shape) * 0.0
        gs0 = int(observation.shape[0] / self.current_grid_map.shape[0])
        gs1 = int(observation.shape[1] / self.current_grid_map.shape[1])
        for i in range(self.current_grid_map.shape[0]):
            for j in range(self.current_grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.current_grid_map[i, j]][k]
                    observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        return (255 * observation).astype(np.uint8)


if __name__=="__main__":
    env = GridworldEnv(plan=1, epislon=0.2, max_range=3, seed=1)

    s = env.reset()
    print(f'initial state: {s}')
    for _ in range(1):
        a = 3
        observation, reward, reached_goal, info = env.step(a)

        print(f'observation: {observation}')
        print(f'reward: {reward}')
        print(f'reached_goal: {reached_goal}')
        print(f'info: {info}')