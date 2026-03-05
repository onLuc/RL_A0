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
        for i in range(n):
            G_t +=(
                    self.gamma ** i * rewards[i] +
                    self.gamma ** n * np.max(self.Q_sa[n]))

        s, a = states[0], actions[0]
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (G_t - self.Q_sa[s, a])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=True)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    states = []
    rewards = []
    actions = []

    for episode in range(n_timesteps):
        done = False
        if episode % eval_interval == 0:
            mean_return = pi.evaluate(env)
            eval_returns.append(mean_return)
            eval_timesteps.append(episode)
        s = env.reset()
        while not done:
            for step in range(max_episode_length):
                action = pi.select_action(s, policy, epsilon, temp)
                s_next, r, done = env.step(action)
                states.append(s)
                actions.append(action)
                rewards.append(r)
                s = s_next
                pi.update(states, actions, rewards, done, n)

    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
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
