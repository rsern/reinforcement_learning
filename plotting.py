"""Drawing helpers for the RL workshop.

You don't need to read this file to follow the workshop. It only draws pictures.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ARROWS = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}  # up, down, left, right
COLORS = {"S": "#dbeafe", "G": "#bbf7d0", "X": "#fecaca", "#": "#9ca3af", ".": "#ffffff"}


def _draw_grid(ax, layout):
    rows, cols = len(layout), len(layout[0])
    for r in range(rows):
        for c in range(cols):
            cell = layout[r][c]
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=COLORS[cell],
                                   edgecolor="#374151", linewidth=1))
            label = {"S": "Start", "G": "Goal", "X": "Pit", "#": "Wall"}.get(cell)
            if label:
                ax.text(c, r + 0.35, label, ha="center", va="center", fontsize=8, color="#374151")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def show_world(env, title="The grid world"):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _draw_grid(ax, env.layout)
    ax.set_title(title)
    plt.show()


def show_path(env, path, title="Agent's path"):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _draw_grid(ax, env.layout)
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, "-o", color="#2563eb", linewidth=2, markersize=5, alpha=0.7)
    ax.set_title(f"{title} ({len(path) - 1} steps)")
    plt.show()


def show_policy(env, Q, title="What the agent learned"):
    """Arrows = best action in each square. Color = how good the square is (max Q)."""
    rows, cols = env.rows, env.cols
    values = np.full((rows, cols), np.nan)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for r in range(rows):
        for c in range(cols):
            if env.layout[r][c] in ".S":
                values[r, c] = Q[env.to_state((r, c))].max()
    im = ax.imshow(values, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Best Q-value (expected future reward)")
    for r in range(rows):
        for c in range(cols):
            cell = env.layout[r][c]
            if cell in ".S":
                a = int(np.argmax(Q[env.to_state((r, c))]))
                dx, dy = ARROWS[a]
                ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.18, head_length=0.12,
                         color="white", length_includes_head=True)
            else:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=COLORS[cell]))
                ax.text(c, r, {"G": "Goal", "X": "Pit", "#": "Wall"}[cell],
                        ha="center", va="center", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.show()


def show_progress(env, snapshots, title="How the Q-table fills in during training"):
    """snapshots: dict of episode number -> copy of Q at that point.
    Same color scale on every panel. No arrow = all four values still tied (untouched)."""
    rows, cols = env.rows, env.cols
    grids = {}
    for ep, Q in snapshots.items():
        values = np.full((rows, cols), np.nan)
        for r in range(rows):
            for c in range(cols):
                if env.layout[r][c] in ".S":
                    values[r, c] = Q[env.to_state((r, c))].max()
        grids[ep] = values
    vmin = min(np.nanmin(v) for v in grids.values())
    vmax = max(np.nanmax(v) for v in grids.values())

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.6 * len(snapshots), 4))
    for ax, (ep, Q) in zip(np.atleast_1d(axes), snapshots.items()):
        im = ax.imshow(grids[ep], cmap="viridis", vmin=vmin, vmax=vmax)
        for r in range(rows):
            for c in range(cols):
                cell = env.layout[r][c]
                if cell in ".S":
                    q = Q[env.to_state((r, c))]
                    if np.all(q == q[0]):
                        continue  # never updated: no opinion yet
                    dx, dy = ARROWS[int(np.argmax(q))]
                    ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.18, head_length=0.12,
                             color="white", length_includes_head=True)
                else:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=COLORS[cell]))
                    ax.text(c, r, {"G": "Goal", "X": "Pit", "#": "Wall"}[cell],
                            ha="center", va="center", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"After {ep} episode{'s' if ep != 1 else ''}")
    fig.colorbar(im, ax=axes, label="Best Q-value", shrink=0.8)
    fig.suptitle(title)
    plt.show()


def plot_rewards(rewards, window=25, title="Total reward per episode"):
    rewards = np.asarray(rewards)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rewards, color="#9ca3af", linewidth=0.8, label="each episode")
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smooth, color="#2563eb", linewidth=2,
                label=f"{window}-episode average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def compare_runs(results, window=25, title="Comparing settings"):
    """results: dict of label -> list of rewards."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, rewards in results.items():
        rewards = np.asarray(rewards)
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smooth, linewidth=2, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-episode avg)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
