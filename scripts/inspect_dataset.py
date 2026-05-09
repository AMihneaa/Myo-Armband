import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_trial(emg: np.ndarray, imu: np.ndarray, title: str) -> None:
    fig, axes= plt.subplots(10, 1, figsize=(14, 20))
    fig.suptitle(title, fontsize=14)

    for ch in range(8):
        axes[ch].plot(emg[:, ch], linewidth=0.8)
        axes[ch].set_ylabel(f"ch{ch+1}", fontsize=8)
        axes[ch].grid(True)

    labels= ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    colors= ['r', 'g', 'b', 'r', 'g', 'b']
    for i in range(3):
        axes[8].plot(imu[:, i], color=colors[i], label=labels[i], linewidth=0.8)
        axes[9].plot(imu[:, i+3], color=colors[i+3], label=labels[i+3], linewidth=0.8)

    axes[8].set_ylabel("Accel", fontsize=8)
    axes[8].legend(fontsize=7)
    axes[8].grid(True)
    axes[9].set_ylabel("Gyro", fontsize=8)
    axes[9].legend(fontsize=7)
    axes[9].grid(True)
    axes[9].set_xlabel("samples")

    plt.tight_layout()
    plt.show()


def inspect(h5_path: str, gesture: str= None, trial: int= None) -> None:
    with h5py.File(h5_path, 'r') as f:
        for subject_key in f.keys():
            for session_key in f[subject_key].keys():
                session= f[subject_key][session_key]

                for gesture_key in session.keys():
                    if gesture and gesture not in gesture_key:
                        continue

                    for trial_key in session[gesture_key].keys():
                        if trial and f"trial_{trial:02d}" != trial_key:
                            continue

                        emg= session[gesture_key][trial_key]['emg'][:]
                        imu= session[gesture_key][trial_key]['imu'][:]

                        title= f"{subject_key} | {session_key} | {gesture_key} | {trial_key}"
                        print(f"{title} — EMG {emg.shape} IMU {imu.shape}")
                        plot_trial(emg, imu, title)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--gesture", type=str, default=None)
    parser.add_argument("--trial", type=int, default=None)
    args= parser.parse_args()

    inspect(args.file, args.gesture, args.trial)

'''

python scripts/inspect_dataset.py --file data_raw/session_1/subject_01_session_1.h5 --gesture hand_close
'''