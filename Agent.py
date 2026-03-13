#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax

import numpy as np
from Helper import softmax, argmax

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, rng=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        if policy == 'greedy':
            return argmax(self.Q_sa[s], rng=self.rng)

        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            max_action = argmax(self.Q_sa[s], rng=self.rng)
            chance = 1 - epsilon * ((self.n_actions - 1) / self.n_actions)

            if chance > self.rng.random():
                a = max_action
            else:
                options_left = list(range(self.n_actions))
                options_left.remove(max_action)
                a = self.rng.choice(options_left)

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            action_probs = softmax(self.Q_sa[s], temp)
            a = self.rng.choice(self.n_actions, p=action_probs)

        return a
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
