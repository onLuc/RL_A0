#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import Helper

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s '''
        a = argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        sum_s_prime = 0
        for s_prime in range(self.n_states):
            s_prime = int(s_prime)
            update = p_sas[s_prime] * (r_sas[s_prime] + self.gamma * np.max(self.Q_sa[s_prime]))
            sum_s_prime += update

        self.Q_sa[s][a] = sum_s_prime
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    episodes = 10000
    for ep in range(episodes):
        delta = 0
        for state in range(QIagent.n_states):
            for action in range(QIagent.n_actions):
                x = QIagent.Q_sa[state][action]
                p_sas, r_sas = env.model(state, action)
                QIagent.update(state,action,p_sas,r_sas)
                delta = max(delta, abs(x-QIagent.Q_sa[state][action]))
        if delta < threshold:
            break
        
    # Plot current Q-value estimates & print max error
    env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()
