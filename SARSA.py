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
        max_a = np.max(self.Q_sa[s_next])
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (
                        r + self.gamma * self.Q_sa[s_next, a_next] - self.Q_sa[s, a])


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=True)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    episodes = 10000
    # TO DO: Write your SARSA algorithm here!
    for episode in range(episodes):
        done = False
        if episode % eval_interval == 0:
            mean_return = pi.evaluate(env)
            eval_returns.append(mean_return)
            eval_timesteps.append(episode)
        s = env.reset()
        while not done:
            action = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(action)
            a_next = pi.select_action(s_next, policy, epsilon, temp)
            pi.update(s, action, r, s_next, a_next, done)
            s = s_next

    if plot:
        s = eval_env.reset()
        for t in range(n_timesteps):
            a = pi.select_action(s, policy="greedy")  # sample random action
            s_next, r, done = eval_env.step(a)  # execute action in the environment
            p_sas, r_sas = eval_env.model(s, a)
            print(
                "State {}, Action {}, Reward {}, Next state {}, Done {}, p(s'|s,a) {}, r(s,a,s') {}".format(
                    s, a, r, s_next, done, p_sas, r_sas))
            eval_env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.5)  # display the environment
            if done:
                s = eval_env.reset()
                quit()
            else:
                s = s_next

    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

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
