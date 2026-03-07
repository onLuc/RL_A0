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
        for i in reversed(range(len(actions))):
            G_t = rewards[i] + self.gamma * G_t
            s, a = states[i], actions[i]
            self.Q_sa[s, a] += self.learning_rate * (G_t - self.Q_sa[s, a])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=True)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)

    eval_timesteps = np.arange(0, n_timesteps, eval_interval)
    eval_returns = []

    successes = 0

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

            while next_eval_idx < len(eval_timesteps) and t >= eval_timesteps[next_eval_idx]:
                eval_returns.append(pi.evaluate(eval_env))
                next_eval_idx += 1
            
            if done:
                successes += 1

        pi.update(states, actions, rewards)
    
    print(f"Successes: {successes}")

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