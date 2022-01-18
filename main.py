import copy

import numpy as np
import random
import matplotlib.pyplot as plt

class Game:
    def __init__(self):
        self.mogura_type = {
                "A": 0,
                "B": 1,
                "C": 2,
        }
        self.actions = {
            "A": 0,
            "B": 1,
            "C": 2,
        }
        self.mogura_area = "B"
    def next_step(self, mogura_area):
        if mogura_area == 0:
            return 1
        elif mogura_area == 2:
            return 1
        else:
            if random.random() < 0.5:
                return 0
            else:
                return 2
    def step(self, action):
        next_area = self.next_step(self.mogura_area)
        reward = 0
        if action == next_area:
            reward = 1
        self.mogura_area = next_area
        return self.mogura_area, reward
    def reset(self):
        self.mogura_area = "B"
        return self.mogura_area


        
class QLearningAgent:

    def __init__(
            self,
            alpha=.2,
            epsilon=.1,
            gamma=.99,
            actions=None,
            observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._init_q_values()

    def _init_q_values(self):
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:
            action = np.argmax(self.q_values[self.state])

        self.previous_action = action
        return action

    def observe(self, next_state, reward=None):
        next_state = str(next_state)
        if next_state not in self.q_values:
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        q = self.q_values[self.previous_state][self.previous_action] 
        max_q = max(self.q_values[self.state]) 
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (reward + (self.gamma * max_q) - q))

NB_EPISODE = 100
EPSILON = .1 
ALPHA = .1  
GAMMA = .90  
ACTIONS = np.arange(3)  

if __name__ == '__main__':
    grid_env = Game()  
    ini_state = grid_env.mogura_area 
    agent = QLearningAgent(
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON, 
        actions=ACTIONS,  
        observation=ini_state) 
    rewards = [] 
    for episode in range(NB_EPISODE):
        episode_reward = []
        i = 0
        while(i < 100):
            action = agent.act()
            state, reward = grid_env.step(action)
            agent.observe(state, reward) 
            episode_reward.append(reward)
            i = i + 1
        rewards.append(np.sum(episode_reward))
        state = grid_env.reset()
        agent.observe(state)
        is_end_episode = False

    plt.plot(np.arange(NB_EPISODE), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("result.jpg")
    plt.show()
