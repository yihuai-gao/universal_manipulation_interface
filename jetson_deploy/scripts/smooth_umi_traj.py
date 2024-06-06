import pickle
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, medfilt
import matplotlib.pyplot as plt
VISUALIZE = True

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import Optional
import seaborn as sns
from rich.progress import track

def smooth_traj(y: np.ndarray, b, a, pad: int = 100, shift_idx: int = 12):
    # smoothed = lfilter(b, a, np.concatenate(([y[0]] * pad, y, [y[-1]] * pad), axis=0))
    # # shift it back
    # smoothed = np.concatenate([smoothed[shift_idx:], [smoothed[-1]] * shift_idx])[
    #     pad:-pad
    # ]
    # median filter, then butter
    smoothed = medfilt(y, 5)
    smoothed = filtfilt(b, a, y, axis=0)
    return smoothed

def visualize(raw, smoothed, path: Optional[str] = None):
    fig, axes = plt.subplots(3, figsize=(15, 15))
    x = np.arange(len(raw))
    for i in range(3):
        sns.lineplot(
            x=x,
            y=raw[:, i],
            ax=axes[i],
            label=f"raw",
        )
        sns.lineplot(
            x=x,
            y=smoothed[:, i],
            ax=axes[i],
            label=f"smoothed",
        )
        axes[i].legend()
        axes[i].set_title(f"Dimension {i}")
    if path is not None:
        plt.tight_layout(pad=0)
        plt.savefig(path)
    else:
        plt.show()
    # close everything
    plt.cla()
    plt.clf()
    plt.close()
    # delete
    del fig
    del axes

VISUALIZE = False

if __name__ == "__main__":
    path = "data/20240605_tossing1_smooth/dataset_plan_raw.pkl"
    # path = "data/20240531_tossing2_smooth/dataset_plan_raw.pkl"
    plan = pickle.load(open(path, "rb"))


    sampling_rate = 60
    pose_order = 2
    pose_cutoff = 4.0
    pose_b, pose_a = butter(
        pose_order, pose_cutoff, fs=sampling_rate, btype="low", analog=False
    )

    for i in track(range(len(plan))):
        ee_pose = plan[i]["grippers"][0]["tcp_pose"]
        t = plan[i]["episode_timestamps"]
        
        dt = np.median(np.diff(t))
        # print(dt)
        # assert dt < 1 / sampling_rate - 1e-2 or dt > 1 / sampling_rate + 1e-2
        ee_pos = ee_pose[:, :3].copy()
        smoothed_ee_pos = np.array(
            [
                smooth_traj(
                    ee_pos[:, i].copy(), b=pose_b, a=pose_a, pad=100, shift_idx=7
                )
                for i in range(3)
            ]
        ).T

        if VISUALIZE:
            visualize(
                ee_pos,
                smoothed_ee_pos,
                path=path.replace("_raw.pkl", f"_pos_{i:04d}.png"),
            )
        ee_pose[:, :3] = ee_pos

        assert np.all(ee_pose == plan[i]["grippers"][0]["tcp_pose"])
    pickle.dump(plan, open(path.replace("raw", "smoothed"), "wb")) # save the smoothed plan