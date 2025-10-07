import os

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Visualize robot position over time.")
parser.add_argument("--logdir", type=str, default=None, help="The log directory containing the npy files.")


def plot_robot_positions(positions, times, death_status):
    survived_env_idx = np.where(np.logical_not(death_status[:, 0, :].any(axis=1)))[0]
    if survived_env_idx.size == 0:
        return

    time = times.squeeze(1)[survived_env_idx[0]]
    positions = np.mean(positions[survived_env_idx], axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    components = ['x', 'y', 'z']  # Common labels for commands and positions

    for i, component in enumerate(components):
        ax = axes[i]  # Get the current subplot

        ax.plot(time, positions[i], label=r"$\mathcal{v}_{" + component + "}$")  # Position

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title(f'Base position in {component}-axis')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Base position over time", fontsize=16)
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()

    try:
        commands = np.load(os.path.join(args.logdir, "command.npy"))[..., :-1]
        times = np.load(os.path.join(args.logdir, "time.npy"))[..., :-1]

        positions = np.load(os.path.join(args.logdir, "base_position.npy"))[..., :-1]
        death_status = np.load(os.path.join(args.logdir, "death_status.npy"))[..., :-1]
    except FileNotFoundError:
        print(f"Error: failed finding one or more files.")
        raise

    plot_robot_positions(positions, times, death_status)
