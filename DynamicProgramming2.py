#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    """Class to store the Q-value iteration solution, perform updates, and select the greedy action."""

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        """Returns the greedy best action in state s."""
        a = argmax(self.Q_sa[s])
        return a

    def update(self, s, a, p_sas, r_sas):
        """Function updates Q(s,a) using p_sas and r_sas."""
        updated_q = 0.0
        for s_prime in range(self.n_states):
            updated_q += p_sas[s_prime] * (
                r_sas[s_prime] + self.gamma * np.max(self.Q_sa[s_prime])
            )
        self.Q_sa[s, a] = updated_q


def plot_q_snapshot(ax, env, Q_sa, title=""):
    """Draw a static Q-value snapshot on the given axis."""
    ax.set_xlim([0, env.width])
    ax.set_ylim([0, env.height])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    # Grid cells and wind shading
    for x in range(env.width):
        for y in range(env.height):
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    linewidth=0,
                    facecolor="k",
                    alpha=env.winds[x] / 4,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    linewidth=0.5,
                    edgecolor="k",
                    fill=False,
                )
            )

    # Outer border
    ax.axvline(0, 0, env.height, linewidth=3, c="k")
    ax.axvline(env.width, 0, env.height, linewidth=3, c="k")
    ax.axhline(0, 0, env.width, linewidth=3, c="k")
    ax.axhline(env.height, 0, env.width, linewidth=3, c="k")

    # Start state
    ax.add_patch(
        Rectangle(env.start_location, 1.0, 1.0, linewidth=0, facecolor="b", alpha=0.2)
    )
    ax.text(
        env.start_location[0] + 0.05,
        env.start_location[1] + 0.75,
        "S",
        fontsize=16,
        c="b",
    )

    # Goal states
    for i, goal in enumerate(env.goal_locations):
        reward = env.goal_rewards[i]
        colour = "g" if reward >= 0 else "r"
        text = f"+{reward}" if reward >= 0 else f"{reward}"
        ax.add_patch(
            Rectangle(goal, 1.0, 1.0, linewidth=0, facecolor=colour, alpha=0.2)
        )
        ax.text(goal[0] + 0.05, goal[1] + 0.75, text, fontsize=16, c=colour)

    # Q-values
    for state in range(env.n_states):
        state_location = env._state_to_location(state)
        for action in range(env.n_actions):
            plot_location = (
                np.array(state_location)
                + 0.42
                + 0.35 * np.array(env.action_effects[action])
            )
            ax.text(
                plot_location[0],
                plot_location[1] + 0.03,
                f"{Q_sa[state, action]:.1f}",
                fontsize=6,
            )

    # Greedy arrows
    for state in range(env.n_states):
        plot_location = np.array(env._state_to_location(state)) + 0.5
        max_actions = np.where(Q_sa[state] == np.max(Q_sa[state]))[0]
        for a in max_actions:
            ax.add_patch(
                Arrow(
                    plot_location[0],
                    plot_location[1],
                    env.action_effects[a][0] * 0.2,
                    env.action_effects[a][1] * 0.2,
                    width=0.05,
                    color="k",
                )
            )


def Q_value_iteration(
    env,
    gamma=1.0,
    threshold=0.001,
    max_iterations=10000,
    render_each_sweep=False,
    snapshot_sweeps=None,
):
    """Runs Q-value iteration. Returns a converged QValueIterationAgent object, deltas, and snapshots."""
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    deltas = []
    snapshots = {}

    if snapshot_sweeps is None:
        snapshot_sweeps = []

    for iteration in range(max_iterations):
        delta = 0.0

        for state in range(QIagent.n_states):
            for action in range(QIagent.n_actions):
                old_q = QIagent.Q_sa[state, action]
                p_sas, r_sas = env.model(state, action)
                QIagent.update(state, action, p_sas, r_sas)
                delta = max(delta, abs(old_q - QIagent.Q_sa[state, action]))

        deltas.append(delta)

        print(f"Sweep {iteration + 1}: max absolute error = {delta:.6f}")

        if render_each_sweep:
            env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)

        if (iteration + 1) in snapshot_sweeps:
            snapshots[iteration + 1] = QIagent.Q_sa.copy()

        if delta < threshold:
            break

    snapshots["converged"] = QIagent.Q_sa.copy()
    return QIagent, deltas, snapshots


def save_dp_progress_figure(env, snapshots, out_file="dp_qvalue_progression.png"):
    """Create a 3-panel figure showing early, intermediate, and converged Q-values."""
    numeric_sweeps = sorted([k for k in snapshots.keys() if isinstance(k, int)])
    if len(numeric_sweeps) < 2:
        raise ValueError("Need at least two numeric snapshots to create the progression figure.")

    early = numeric_sweeps[0]
    mid = numeric_sweeps[len(numeric_sweeps) // 2]
    conv = "converged"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    plot_q_snapshot(axes[0], env, snapshots[early], title=f"After sweep {early}")
    plot_q_snapshot(axes[1], env, snapshots[mid], title=f"After sweep {mid}")
    plot_q_snapshot(axes[2], env, snapshots[conv], title="Converged")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)

    # Save snapshots for the report figure
    snapshot_sweeps = [1, 5, 10]

    # Goal A: default goal
    QIagent1, deltas1, snapshots1 = Q_value_iteration(
        env,
        gamma,
        threshold,
        max_iterations=10000,
        render_each_sweep=False,
        snapshot_sweeps=snapshot_sweeps,
    )

    start_state = env.reset()
    V1 = float(np.max(QIagent1.Q_sa[start_state]))
    print("V*(s=3) =", V1)

    save_dp_progress_figure(env, snapshots1, out_file="dp_qvalue_progression.png")

    # Optional rollout for default goal
    done = False
    s = env.reset()
    for _ in range(500):
        a = QIagent1.select_action(s)
        s, r, done = env.step(a)
        env.render(Q_sa=QIagent1.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        if done:
            break

    # Goal B: changed goal [6,2]
    env.goal_locations = [[6, 2]]
    env.goal_rewards = [100]
    env._construct_model()

    QIagent2, deltas2, _ = Q_value_iteration(
        env,
        gamma,
        threshold,
        max_iterations=10000,
        render_each_sweep=False,
        snapshot_sweeps=[],
    )

    start_state = env.reset()
    V2 = float(np.max(QIagent2.Q_sa[start_state]))
    print("V*(s=3) with goal=[6,2] =", V2)

    # Optional rollout for changed goal
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

    snapshot_sweeps = [1, 5, 10]

    QIagent1, deltas1, snapshots1 = Q_value_iteration(
        env,
        gamma,
        threshold,
        max_iterations=10000,
        render_each_sweep=False,
        snapshot_sweeps=snapshot_sweeps,
    )
    start_state = env.reset()
    V1 = float(np.max(QIagent1.Q_sa[start_state]))
    print("V*(s=3) =", V1)

    save_dp_progress_figure(env, snapshots1, out_file="dp_qvalue_progression.png")

    env.goal_locations = [[6, 2]]
    env.goal_rewards = [100]
    env._construct_model()

    QIagent2, deltas2, _ = Q_value_iteration(
        env,
        gamma,
        threshold,
        max_iterations=10000,
        render_each_sweep=False,
        snapshot_sweeps=[],
    )
    start_state = env.reset()
    V2 = float(np.max(QIagent2.Q_sa[start_state]))
    print("V*(s=3) with goal=[6,2] =", V2)

    plt.figure(figsize=(7, 4))
    plt.plot(deltas1, label=f"Goal [7,3]  (V*(s=3)={V1:.2f}, sweeps={len(deltas1)})")
    plt.plot(deltas2, label=f"Goal [6,2]  (V*(s=3)={V2:.2f}, sweeps={len(deltas2)})")
    plt.yscale("log")
    plt.xlabel("Sweep")
    plt.ylabel("Max absolute error δ")
    plt.title("DP convergence comparison for two goal locations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dp_convergence_two_goals.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    experiment_old()