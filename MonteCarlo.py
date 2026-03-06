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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        G_t = 0
        for i in range(len(rewards)):
            G_t += self.gamma ** i * rewards[i]

        s, a = states[0], actions[0]
        self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (G_t - self.Q_sa[s, a])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=True)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    for episode in range(n_timesteps):

        if episode % eval_interval == 0:
            mean_return = pi.evaluate(env)
            print(mean_return)
            eval_returns.append(mean_return)
            eval_timesteps.append(episode)

        s = env.reset()
        states = [s]
        rewards = []
        actions = []
        done = False

        step = 0
        while not done and step < max_episode_length:
            # for step in range(max_episode_length):
            action = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(action)
            states.append(s_next)
            actions.append(action)
            rewards.append(r)
            s = s_next

        for i in range(len(rewards)):
            pi.update(states[i:], actions[i:], rewards[i:])

    if plot:
        s = eval_env.reset()
        for t in range(n_timesteps):
            a = pi.select_action(s, policy="greedy")  # sample random action
            s_next, r, done = eval_env.step(
                a)  # execute action in the environment
            p_sas, r_sas = eval_env.model(s, a)
            print(
                "State {}, Action {}, Reward {}, Next state {}, Done {}, p(s'|s,a) {}, r(s,a,s') {}".format(
                    s, a, r, s_next, done, p_sas, r_sas))
            eval_env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                            step_pause=0.5)  # display the environment
            if done:
                s = eval_env.reset()
                quit()
            else:
                s = s_next
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
