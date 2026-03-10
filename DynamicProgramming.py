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
import matplotlib.pyplot as plt

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))

    def select_action(self, s):
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
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001, max_iterations=10000, render_each_sweep=False):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    # iteration = 0
    # while True:
    deltas = []
    for iteration in range(max_iterations):
        delta = 0.0

        for state in range(QIagent.n_states):
            for action in range(QIagent.n_actions):
                x = QIagent.Q_sa[state][action]
                p_sas, r_sas = env.model(state, action)
                QIagent.update(state, action, p_sas, r_sas)
                delta = max(delta, abs(x - QIagent.Q_sa[state][action]))

        deltas.append(delta)

        # Plot current Q-value estimates & print max error
        print(f"Sweep {iteration}: max absolute error = {delta:.6f}")
        if render_each_sweep:
            env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)

        if delta < threshold:
            break
        # iteration += 1

    return QIagent, deltas

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)

    # --- Goal A: default goal ---
    QIagent = Q_value_iteration(env, gamma, threshold, max_iterations=10000, render_each_sweep=False)

    start_state = env.reset()
    V_start = np.max(QIagent.Q_sa[start_state])
    print("V*(s=3) =", V_start)

    # Rollout for goal A (cap steps)
    done = False
    s = env.reset()
    for _ in range(500):
        a = QIagent.select_action(s)
        s, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        if done:
            break

    # --- Goal B: changed goal [6,2] ---
    env.goal_locations = [[6, 2]]
    env.goal_rewards = [100]
    env._construct_model()

    QIagent2 = Q_value_iteration(env, gamma, threshold, max_iterations=10000, render_each_sweep=False)

    start_state = env.reset()
    V_start2 = np.max(QIagent2.Q_sa[start_state])
    print("V*(s=3) with goal=[6,2] =", V_start2)

    # Rollout for goal B (cap steps)
    done = False
    s = env.reset()
    for _ in range(500):
        a = QIagent2.select_action(s)
        s, r, done = env.step(a)
        env.render(Q_sa=QIagent2.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        if done:
            break

def experiment_old():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)

    QIagent1, deltas1 = Q_value_iteration(env, gamma, threshold, max_iterations=10000, render_each_sweep=False)
    start_state = env.reset()
    V1 = float(np.max(QIagent1.Q_sa[start_state]))
    print("V*(s=3) =", V1)

    env.goal_locations = [[6, 2]]
    env.goal_rewards = [100]
    env._construct_model()

    QIagent2, deltas2 = Q_value_iteration(env, gamma, threshold, max_iterations=10000, render_each_sweep=False)
    start_state = env.reset()
    V2 = float(np.max(QIagent2.Q_sa[start_state]))
    print("V*(s=3) with goal=[6,2] =", V2)

    plt.figure(figsize=(7, 4))
    plt.plot(deltas1, label=f"Goal [7,3]  (V*(s=3)={V1:.2f}, sweeps={len(deltas1)})")
    plt.plot(deltas2, label=f"Goal [6,2]  (V*(s=3)={V2:.2f}, sweeps={len(deltas2)})")
    plt.yscale("log")  # makes convergence comparison much clearer
    plt.xlabel("Sweep")
    plt.ylabel("Max absolute error δ")
    plt.title("DP convergence comparison for two goal locations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dp_convergence_two_goals.png", dpi=200)
    # plt.show()

    # env.goal_locations = [[7, 3]]
    # env.goal_rewards = [100]
    # env._construct_model()
    # env.render(Q_sa=QIagent1.Q_sa, plot_optimal_policy=True, step_pause=0.5)
    #
    # env.goal_locations = [[6, 2]]
    # env.goal_rewards = [100]
    # env._construct_model()
    # env.render(Q_sa=QIagent2.Q_sa, plot_optimal_policy=True, step_pause=0.5)
    
if __name__ == '__main__':
    experiment_old()
