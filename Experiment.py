#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time

from Q_learning import q_learning
from SARSA import sarsa
from Nstep import n_step_Q
from MonteCarlo import monte_carlo
from Helper import LearningCurvePlot, smooth


def mean_ci(curves, ci=0.95):
    curves = np.asarray(curves)
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0, ddof=1)
    n = curves.shape[0]
    se = std / np.sqrt(n)

    z = 1.96 if ci == 0.95 else 1.0
    half = z * se
    return mean, mean - half, mean + half


def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma,
                             policy='egreedy',
                             epsilon=None, temp=None, smoothing_window=None, plot=False, n=5, eval_interval=500,
                             goal_locations=None, goal_rewards=None):
    returns_over_repetitions = []
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        if backup == 'q':
            returns, timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot,
                                            eval_interval, goal_locations, goal_rewards)
        elif backup == 'sarsa':
            returns, timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
        elif backup == 'nstep':
            returns, timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                                          policy, epsilon, temp, plot, n, eval_interval)
        elif backup == 'mc':
            returns, timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                                             policy, epsilon, temp, plot, eval_interval)

        returns_over_repetitions.append(returns)

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    curves = np.asarray(returns_over_repetitions)  # (n_reps, T)
    mean, low, high = mean_ci(curves, ci=0.95)

    if smoothing_window is not None:
        mean = smooth(mean, smoothing_window)
        low = smooth(low, smoothing_window)
        high = smooth(high, smoothing_window)

    return mean, low, high, timesteps


def experiment():
    ####### Settings
    # Experiment      
    n_repetitions = 20  # 20
    smoothing_window = 5  # 9 # Must be an odd number. Use 'None' to switch smoothing off!
    plot = False  # Plotting is very slow, switch it off when we run repetitions

    # MDP    
    n_timesteps = 50001  # 50001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 1000  # 1000
    max_episode_length = 100
    gamma = 1.0

    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.05
    temp = 1.0
    # Back-up & update
    backup = 'q'  # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.1
    n = 5  # only used when backup = 'nstep'

    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                     'sarsa': 'SARSA',
                     'mc': 'Monte Carlo',
                     'nstep': 'n-step Q-learning'}

    ####### Experiments

    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_episode_return = 83.67825468739042  # set the optimal return per episode you found in the DP assignment here

    #### Assignment 2: Effect of exploration
    # policy = 'egreedy'
    # epsilons = [0.03, 0.1, 0.3]
    # learning_rate = 0.1
    # backup = 'q'
    # # Added the r before the string as \e is apparently an invalid escape statement
    # Plot = LearningCurvePlot(title=r'Exploration: $\epsilon$-greedy versus softmax exploration')
    # Plot.set_ylim(-100, 100)
    # for epsilon in epsilons:
    #     print('Running {}-greedy with epsilon = {}'.format(backup, epsilon))
    #     mean, low, high, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length,
    #                                                           learning_rate,
    #                                                           gamma, policy, epsilon, temp, smoothing_window, plot, n,
    #                                                           eval_interval)
    #     Plot.add_curve_with_ci(timesteps, mean, low, high, label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
    # policy = 'softmax'
    # temps = [0.01, 0.1, 1.0]
    # for temp in temps:
    #     print('Running {} with softmax temperature = {}'.format(backup, temp))
    #     mean, low, high, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length,
    #                                                           learning_rate,
    #                                                           gamma, policy, epsilon, temp, smoothing_window, plot, n,
    #                                                           eval_interval)
    #     Plot.add_curve_with_ci(timesteps, mean, low, high, label=r'softmax, $ \tau $ = {}'.format(temp))
    # Plot.add_hline(optimal_episode_return, label="DP optimum")
    # Plot.save('exploration.png')

    # ###### Assignment 3: Q-learning versus SARSA
    policy = 'egreedy'
    epsilon = 0.1  # set epsilon back to original value
    learning_rates = [0.03, 0.1, 0.3]
    backups = ['q', 'sarsa']
    Plot = LearningCurvePlot(title='Back-up: on-policy versus off-policy')
    Plot.set_ylim(-100, 100)
    for backup in backups:
        for learning_rate in learning_rates:
            print('Running {} with learning rate = {}'.format(backup, learning_rate))
            mean, low, high, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps,
                                                                  max_episode_length, learning_rate,
                                                                  gamma, policy, epsilon, temp, smoothing_window, plot,
                                                                  n, eval_interval)
            Plot.add_curve_with_ci(timesteps, mean, low, high,
                                   label=r'{}, $\alpha$ = {} '.format(backup_labels[backup], learning_rate))
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save('on_off_policy.png')

    ##### Assignment 4: Back-up depth
    policy = 'egreedy'
    epsilon = 0.05  # set epsilon back to original value
    learning_rate = 0.1
    backup = 'nstep'
    ns = [1, 3, 10]
    Plot = LearningCurvePlot(title='Back-up: depth')
    Plot.set_ylim(-100, 100)
    for n in ns:
        print('Running {} with n = {}'.format(backup, n))
        mean, low, high, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length,
                                                              learning_rate,
                                                              gamma, policy, epsilon, temp, smoothing_window, plot, n,
                                                              eval_interval)
        Plot.add_curve_with_ci(timesteps, mean, low, high, label=r'{}-step Q-learning'.format(n))

    backup = 'mc'

    # Keeping the same parameters and commenting these out for a fair comparison, or not!
    # MC parameters:
    epsilon = 0.2  # 0.4, 0.2
    learning_rate = 0.03  # 0.02, 0.03
    max_episode_length = 300  # 300

    print('Running Monte Carlo')
    mean, low, high, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length,
                                                         learning_rate,
                                                         gamma, policy, epsilon, temp, smoothing_window, plot, n,
                                                         eval_interval)
    Plot.add_curve_with_ci(timesteps, mean, low, high, label='Monte Carlo')
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save('depth.png')


if __name__ == '__main__':
    experiment()
