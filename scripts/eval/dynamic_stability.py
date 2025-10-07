import os

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Visualize robot velocities.")
parser.add_argument("--logdir", type=str, default=None, help="The log directory containing the npy files.")
parser.add_argument("--push_time", type=float, default=5.0, help="The time at which the push was applied")


def compute_recovery_times(commands, velocities, death_status, times, push_time=5.0, push_axis=1, epsilon=0.15):
    max_episode_len = commands.shape[2]
    recovery_times = []

    survived_env_idx = np.where(np.logical_not(death_status[:, 0, :].any(axis=1)))[0]

    for env_idx in survived_env_idx:
        command = commands[env_idx]
        velocity = velocities[env_idx]
        time = times[env_idx, 0, :]

        # Find the index closest to the push time
        push_index = np.argmin(np.abs(time - push_time))
        
        command_at_push = command[push_axis, push_index]
        for t in range(push_index + 1, max_episode_len):
            if np.abs(velocity[push_axis, t - 1] - command_at_push) <= epsilon:
                # Recovery found, move on
                recovery_times.append(time[t] - time[push_index])
                break
    
    return survived_env_idx, recovery_times


def compute_push_displacements(positions, commands, velocities, death_status, times, push_time=5.0, push_axis=1, epsilon=0.15):
    push_displacements = []

    survived_env_idx, time_recoveries = compute_recovery_times(commands, velocities, death_status, times, push_time, push_axis, epsilon)

    for env_idx, time_recovery in zip(survived_env_idx, time_recoveries):
        position = positions[env_idx]
        time = times[env_idx, 0, :]

        push_index = np.argmin(np.abs(time - push_time))
        recovered_index = np.argmin(np.abs(time - (push_time + time_recovery)))

        push_displacements.append(np.abs(position[push_axis, push_index] - position[push_axis, recovered_index]))

    return survived_env_idx, push_displacements


if __name__ == "__main__":
    args = parser.parse_args()

    eval_names = []
    eval_names += ["stab_side_s", "stab_side_m", "stab_side_l"]

    time_recoveries = []
    push_displacements = []

    for eval_name in eval_names:
        logdir = os.path.join(args.logdir, "eval_logs", eval_name)
        push_axis = 1 if "side" in eval_name else 0

        # Load npy log files
        try:
            commands = np.load(os.path.join(logdir, "command.npy"))[..., :-1]
            times = np.load(os.path.join(logdir, "time.npy"))[..., :-1]

            death_status = np.load(os.path.join(logdir, "death_status.npy"))[..., :-1]
    
            linear_vels = np.load(os.path.join(logdir, "base_linear_velocity.npy"))[..., :-1]
            positions = np.load(os.path.join(logdir, "base_position.npy"))[..., :-1]
        except FileNotFoundError:
            print(f"Error: failed finding one or more files.")
            raise

        _, per_env_time_recoveries = compute_recovery_times(commands, linear_vels, death_status, times, push_time=args.push_time, push_axis=push_axis)
        time_recoveries.append(np.mean(per_env_time_recoveries))

        _, per_env_push_displacements = compute_push_displacements(positions, commands, linear_vels, death_status, times, push_time=args.push_time, push_axis=push_axis)
        push_displacements.append(np.mean(per_env_push_displacements))

    print("Results:")
    for eval_name, time_recovery, push_displacement in zip(eval_names, time_recoveries, push_displacements):
        if "side" not in eval_name:
            print(f"{eval_name} - time_recovery: {time_recovery:.3f} s")
            continue

        print(f"{eval_name} - time_recovery: {time_recovery:.3f} s - push_displacement: {push_displacement:.3f} m")
