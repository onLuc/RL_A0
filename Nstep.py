#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is a terminal state '''
        # TO DO: Add own code
        G_t = 0
        for i in range(len(rewards)):
            G_t += self.gamma ** i * rewards[i]

        if not done:
            G_t += self.gamma ** len(rewards) * np.max(self.Q_sa[states[-1]])

        s, a = states[0], actions[0]
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (G_t - self.Q_sa[s, a])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500, seed=None):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''


    master_rng = np.random.default_rng(seed)

    env_rng = np.random.default_rng(master_rng.integers(0, 2 ** 32))
    eval_rng = np.random.default_rng(master_rng.integers(0, 2 ** 32))
    agent_rng = np.random.default_rng(master_rng.integers(0, 2 ** 32))

    env = StochasticWindyGridworld(initialize_model=False, rng=env_rng)
    eval_env = StochasticWindyGridworld(initialize_model=True, rng=eval_rng)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, rng=agent_rng)
    
    # env = StochasticWindyGridworld(initialize_model=False)
    # eval_env = StochasticWindyGridworld(initialize_model=True)
    # pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

    eval_timesteps = np.arange(0, n_timesteps, eval_interval)
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    eval_returns.append(pi.evaluate(eval_env))
    next_eval_idx = 1
    t = 0

    while t < n_timesteps:
        s = env.reset()
        states = [s]
        rewards = []
        actions = []
        done = False

        step = 0
        while (not done) and (step < max_episode_length) and (t < n_timesteps):
            action = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(action)

            states.append(s_next)
            actions.append(action)
            rewards.append(r)

            s = s_next
            step += 1
            t += 1

            if len(rewards) == n:
                pi.update(states, actions, rewards, done, n)
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

            while next_eval_idx < len(eval_timesteps) and t >= eval_timesteps[next_eval_idx]:
                eval_returns.append(pi.evaluate(eval_env))
                next_eval_idx += 1
                
        while len(actions) > 0:
            pi.update(states, actions, rewards, done, len(actions))
            states.pop(0)
            actions.pop(0)
            rewards.pop(0)

    if plot:
        s = eval_env.reset()
        for t in range(n_timesteps):
            a = pi.select_action(s, policy="greedy")  # sample random action
            s_next, r, done = eval_env.step(a)  # execute action in the environment
            eval_env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)  # display the environment
            if done:
                s = eval_env.reset()
            else:
                s = s_next

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
