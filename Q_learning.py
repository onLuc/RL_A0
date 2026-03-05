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
        
    def update(self,s,a,r,s_next,done):
        # max_a = -math.inf
        # for action in range(self.n_actions):
        #     a_prime = self.Q_sa[s_next, action]
        #     if a_prime > max_a:
        #         max_a = a_prime
        max_a = np.max(self.Q_sa[s_next])
        G_t = r + self.gamma * max_a
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (G_t - self.Q_sa[s, a])


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []


    episodes = 10000
    # TO DO: Write your Q-learning algorithm here!
    for episode in range(episodes):
        done = False
        if episode % eval_interval == 0:
            mean_return = agent.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(episode)
            print(episode, mean_return)
        s = env.reset()
        while not done:
            action = agent.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(action)
            agent.update(s, action, r, s_next, done)
            s = s_next
    
    if plot:
       env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.5) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 10000
    eval_interval = 100
    # eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns)

if __name__ == '__main__':
    test()
