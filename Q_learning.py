#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
import math

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


class QLearningAgent(BaseAgent):

    def update(self, s, a, r, s_next, done):
        max_a = np.max(self.Q_sa[s_next])
        G_t = r if done else r + self.gamma * max_a
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (G_t - self.Q_sa[s, a])


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True,
               eval_interval=500, goal_locations=None, goal_rewards=None):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=True)

    if goal_locations is not None:
        env.goal_locations = goal_locations
        eval_env.goal_locations = goal_locations
    if goal_rewards is not None:
        env.goal_rewards = goal_rewards
        eval_env.goal_rewards = goal_rewards
    if goal_locations is not None or goal_rewards is not None:
        eval_env._construct_model()

    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Q-learning algorithm here!
    s = env.reset()
    for t in range(n_timesteps):
        if t % eval_interval == 0:
            mean_return = agent.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)

        action = agent.select_action(s, policy, epsilon, temp)
        s_next, r, done = env.step(action)
        agent.update(s, action, r, s_next, done)

        if done:
            s = env.reset()
        else:
            s = s_next

    if plot:
        s = eval_env.reset()
        for t in range(n_timesteps):
            a = agent.select_action(s, policy="greedy")  # sample random action
            s_next, r, done = eval_env.step(a)  # execute action in the environment

            eval_env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.5)  # display the environment

            if done:
                s = eval_env.reset()
            else:
                s = s_next

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000
    eval_interval = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot,
                                              eval_interval)


if __name__ == '__main__':
    test()
