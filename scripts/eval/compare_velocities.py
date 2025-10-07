import os

import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["legend.loc"] = "lower left"

parser = argparse.ArgumentParser(description="Visualize robot velocities.")
parser.add_argument("--baseline", type=str, default=None, help="The log directory containing the baseline variant's npy files.")
parser.add_argument("--perceptive", type=str, default=None, help="The log directory containing the perceptive variant's npy files.")
parser.add_argument("--reflexive", type=str, default=None, help="The log directory containing the reflexive variant's npy files.")
parser.add_argument("--combined", type=str, default=None, help="The log directory containing the combined variant's npy files.")


def plot_robot_velocities(commands, times, baseline, perceptive, reflexive, combined):
    time = times.squeeze(1)[0]

    mean_commands = np.mean(commands, axis=0)
    mean_baseline = np.mean(baseline, axis=0)
    mean_perceptive = np.mean(perceptive, axis=0)
    mean_reflexive = np.mean(reflexive, axis=0)
    mean_combined = np.mean(combined, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.53), sharex=False)  # 3 rows, 1 column, no shared x

    tracking_data = [
        {'index': 0, 'ylabel': "$v_{b,x}$ (m/s)", 'time_range': (0, 10)},
        {'index': 1, 'ylabel': "$v_{b,y}$ (m/s)", 'time_range': (8, 16)},
        {'index': 2, 'ylabel': "$\omega_{b,z}$ (rad/s)", 'time_range': (14, 22)}
    ]

    for i, data in enumerate(tracking_data):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        index = data['index']
        ylabel = data['ylabel']
        time_start, time_end = data['time_range']

        # Find the indices corresponding to the desired time range
        start_index = np.argmin(np.abs(time - time_start))
        end_index = np.argmin(np.abs(time - time_end))

        ax.plot(time[start_index:end_index], mean_commands[index, start_index:end_index], label=r"Command", linewidth=1)
        ax.plot(time[start_index:end_index], mean_baseline[index, start_index:end_index], label=r"Base", color="orange", linewidth=1)
        ax.plot(time[start_index:end_index], mean_perceptive[index, start_index:end_index], label=r"Perceptive", color="teal", linewidth=1)
        ax.plot(time[start_index:end_index], mean_reflexive[index, start_index:end_index], label=r"Postural", color="red", linewidth=1)
        ax.plot(time[start_index:end_index], mean_combined[index, start_index:end_index], label=r"Full", color="darkviolet", linewidth=1)

        ax.set_ylabel(ylabel)

        if i == 2:
            ax.set_xlabel('\small Time (s)')
            ax.legend(bbox_to_anchor=(0.0, -0.6, 1.0, 0.1), loc='upper left', ncols=3, mode="expand", borderaxespad=0., fontsize='small')

        ax.grid(True)

    plt.tight_layout()
    plt.show()
    # Can also use plt.savefig("path") instead of plt.show() (be sure to remove show if using savefig)

if __name__ == "__main__":
    args = parser.parse_args()

    # Load npy log files
    try:
        commands = np.load(os.path.join(args.combined, "command.npy"))[..., 200:-200]
        times = np.load(os.path.join(args.combined, "time.npy"))[..., 200:-200]

        baseline = np.load(os.path.join(args.baseline, "base_tracked_velocity.npy"))[..., 200:-200]
        perceptive = np.load(os.path.join(args.perceptive, "base_tracked_velocity.npy"))[..., 200:-200]
        reflexive = np.load(os.path.join(args.reflexive, "base_tracked_velocity.npy"))[..., 200:-200]
        combined = np.load(os.path.join(args.combined, "base_tracked_velocity.npy"))[..., 200:-200]
    except FileNotFoundError:
        print(f"Error: failed finding one or more files.")
        raise

    plot_robot_velocities(commands, times, baseline, perceptive, reflexive, combined)
