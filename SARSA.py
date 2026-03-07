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

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        if done:
            target = r
        else:
            target = r + self.gamma * self.Q_sa[s_next, a_next]
        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=True)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!
    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp)

    for t in range(n_timesteps):
        if t % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)

        s_next, r, done = env.step(a)

        if done:
            a_next = None
        else:
            a_next = pi.select_action(s_next, policy, epsilon, temp)

        pi.update(s, a, r, s_next, a_next, done)

        if done:
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            s = s_next
            a = a_next

    if plot:
        s = eval_env.reset()
        for t in range(n_timesteps):
            a = pi.select_action(s, policy="greedy")  # sample random action
            s_next, r, done = eval_env.step(a)  # execute action in the environment
            eval_env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.5)  # display the environment
            if done:
                s = eval_env.reset()
            else:
                s = s_next

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
            
    
if __name__ == '__main__':
    test()
