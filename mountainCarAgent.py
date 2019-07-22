import sys
import gym
import numpy as np

import pandas as pd

def create_uniform_grid(low, high, bins=(10, 10)):
    return [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

class QLearningAgent:
    def __init__(self, env, state_grid, alpha=0.02, gamma=0.99, epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid) 
        self.action_size = self.env.action_space.n 
        self.seed = np.random.seed(seed)
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = self.initial_epsilon = epsilon 
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))

    def preprocess_state(self, state):
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        state = self.preprocess_state(state)
        if mode == 'test':
            action = np.argmax(self.q_table[state])
        else:
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                action = np.random.randint(0, self.action_size)
            else:
                action = np.argmax(self.q_table[state])
        self.last_state = state
        self.last_action = action
        return action

if __name__ == '__main__':
    num_episodes=20000
    mode='train'
    env = gym.make('MountainCar-v0')
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
    agent = QLearningAgent(env, state_grid)
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        scores.append(total_reward)
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    state = env.reset()
    score = 0
    for t in range(200):
        action = agent.act(state, mode='test')
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    print('Final score:', score)
    env.close()