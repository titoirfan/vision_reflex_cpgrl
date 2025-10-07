import os

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Visualize robot velocities.")
parser.add_argument("--logdir", type=str, default=None, help="The log directory containing the npy files.")


def plot_robot_velocities(commands, velocities, times, death_status):
    survived_env_idx = np.where(np.logical_not(death_status[:, 0, :].any(axis=1)))[0]
    if survived_env_idx.size == 0:
        return

    time = times.squeeze(1)[survived_env_idx[0]]
    commands = np.mean(commands[survived_env_idx], axis=0)
    velocities = np.mean(velocities[survived_env_idx], axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    components = ['x', 'y', 'z']  # Common labels for commands and velocities

    for i, component in enumerate(components):
        ax = axes[i]  # Get the current subplot

        ax.plot(time, commands[i], label=r"$\mathcal{v}_{" + component + "}$")  # Command
        ax.plot(time, velocities[i], label=r"$\mathcal{v}_{" + component + "}$")  # Velocity

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Base velocity in {component}-axis')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Base velocity tracking over time", fontsize=16)
    plt.show()


def compute_settling_time(commands, velocities, times, death_status, tolerance=0.05):
    survived_env_idx = np.where(np.logical_not(death_status[:, 0, :].any(axis=1)))[0]
    if survived_env_idx.size == 0:
        return -1

    N, _, T = commands.shape
    settling_times = np.full((3,), np.inf)

    times = times.squeeze(1)[survived_env_idx[0]]
    commands = np.mean(commands[survived_env_idx], axis=0)
    velocities = np.mean(velocities[survived_env_idx], axis=0)

    for axis in range(3):
        command = commands[axis, -1]  # Steady-state command (last value)
        upper_bound = command * (1 + tolerance)
        lower_bound = command * (1 - tolerance)

        settled = False
        for t in range(1, T):
            if (lower_bound <= velocities[axis, t] <= upper_bound):
              settling_times[axis] = times[t]
              settled = True
              break
        if not settled:
            print(f"Settling time for axis {axis} not achieved within the time frame.")

    return settling_times


def compute_steady_state_error(commands, velocities, death_status):
    survived_env_idx = np.where(np.logical_not(death_status[:, 0, :].any(axis=1)))[0]
    if survived_env_idx.size == 0:
        return -1
    
    N, _, T = commands.shape

    commands = np.mean(commands[survived_env_idx], axis=0)
    velocities = np.mean(velocities[survived_env_idx], axis=0)

    steady_state_error = np.abs(commands[:, -1] - velocities[:, -1])  # Error at the last timestep

    return steady_state_error


def compute_vel_mae_intended(commands, velocities, death_status):
    survived_env_idx = np.where(np.logical_not(death_status[:, 0, :].any(axis=1)))[0]
    if survived_env_idx.size == 0:
        return -1

    commands_survived = commands[survived_env_idx]
    command_mask = commands_survived != 0
    velocities_survived = velocities[survived_env_idx] * command_mask

    tracking_error = np.linalg.norm((commands_survived - velocities_survived), axis=1)
    mean_tracking_error = np.mean(tracking_error)

    return mean_tracking_error


if __name__ == "__main__":
    args = parser.parse_args()

    try:
        commands = np.load(os.path.join(args.logdir, "command.npy"))[..., :-1]
        times = np.load(os.path.join(args.logdir, "time.npy"))[..., :-1]
        death_status = np.load(os.path.join(args.logdir, "death_status.npy"))[..., :-1]

        velocities = np.load(os.path.join(args.logdir, "base_tracked_velocity.npy"))[..., :-1]
    except FileNotFoundError:
        print(f"Error: failed finding one or more files.")
        raise

    # limits = [0, 304, 604, 904, 1204, 1504, 1804, 2104]
    # settling_times = []
    # steady_state_errors = []

    # for i in range(len(limits) - 1):
    #     command = commands[:, :, limits[i]:limits[i+1]]
    #     velocity = velocities[:, :, limits[i]:limits[i+1]]
    #     time = times[:, :, limits[i]:limits[i+1]]
    #     death_state = death_status[:, :, limits[i]:limits[i+1]]

    #     settling_times.append(compute_settling_time(command, velocity, time, death_state))
    #     steady_state_errors.append(compute_steady_state_error(command, velocity, death_state))

    # print(f"Settling time: {settling_times}")
    # print(f"Steady state error: {steady_state_errors}")

    print(f"Settling time: {compute_settling_time(commands, velocities, times, death_status)}")
    print(f"Steady state error: {compute_steady_state_error(commands, velocities, death_status)}")
    print(f"Velocity tracking MAE: {compute_vel_mae_intended(commands, velocities, death_status)}")

    plot_robot_velocities(commands, velocities, times, death_status)
