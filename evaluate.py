"""
Evaluation script for trained Franka trajectory-tracking policy.

Usage:
    python evaluate.py                             # use models/best_model.zip
    python evaluate.py --model models/final_model  # specify checkpoint

Outputs (saved to results/):
    trajectory_3d.png       Desired vs actual paths for all three trajectories
    error_over_time.png     Per-step tracking error curves
    metrics_table.txt       Mean/max/RMSE error, jerk, success-rate table
    rollout_circle.mp4      Video of the circle trajectory
    rollout_figure8.mp4
    rollout_lissajous.mp4
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401 (registers 3D projection)
import numpy as np
import imageio.v3 as iio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tracking_env import FrankaTrackingEnv, TRAJ_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_eval_env(traj_type: str, noise_sigma: float = 0.0):
    """Single-env wrapper, noise level configurable for robustness testing."""
    def _init():
        env = FrankaTrackingEnv(
            trajectory_type=traj_type,
            add_noise=(noise_sigma > 0),
            render_mode="rgb_array",
        )
        # Override noise sigma if specified
        if noise_sigma > 0:
            env.NOISE_Q  = noise_sigma
            env.NOISE_EE = noise_sigma * 2
        return env
    return DummyVecEnv([_init])


def load_model(model_path: str, vecnorm_path: str, traj_type: str,
               noise_sigma: float = 0.0):
    vec_env = make_eval_env(traj_type, noise_sigma)
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env)
    return model, vec_env


def rollout(model, vec_env, n_episodes: int = 3, capture_video: bool = False):
    """Run n_episodes and return trajectories + metrics."""
    all_ee, all_target, all_errors, all_actions = [], [], [], []
    frames = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ee_traj, target_traj, errors, actions = [], [], [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            done = bool(done_arr[0])
            raw_info = info[0]
            ee_traj.append(raw_info["ee_pos"])
            target_traj.append(raw_info["target_pos"])
            errors.append(raw_info["error_norm"])
            actions.append(action[0].copy())

            if capture_video and ep == 0:
                frame = vec_env.envs[0].render()
                if frame is not None:
                    frames.append(frame)

        all_ee.append(np.array(ee_traj))
        all_target.append(np.array(target_traj))
        all_errors.append(np.array(errors))
        all_actions.append(np.array(actions))

    return all_ee, all_target, all_errors, all_actions, frames


def compute_metrics(all_errors: list, all_actions: list) -> dict:
    errors = np.concatenate(all_errors)
    actions = np.concatenate(all_actions, axis=0)

    # Jerk: second finite difference of actions
    jerks = []
    for acts in all_actions:
        if len(acts) > 2:
            jerks.append(np.linalg.norm(np.diff(acts, n=2, axis=0), axis=1))
    jerk_mean = float(np.mean(np.concatenate(jerks))) if jerks else 0.0

    return {
        "mean_error_m":  float(np.mean(errors)),
        "max_error_m":   float(np.max(errors)),
        "rmse_m":        float(np.sqrt(np.mean(errors ** 2))),
        "success_rate":  float(np.mean(errors < 0.02)),   # <2 cm threshold
        "jerk_mean":     jerk_mean,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_3d_trajectories(traj_results: dict, save_path: str):
    fig = plt.figure(figsize=(16, 5))
    titles = {"circle": "Circle", "figure8": "Figure-Eight", "lissajous": "Lissajous"}

    for idx, traj_type in enumerate(TRAJ_TYPES):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ee_list, target_list = traj_results[traj_type]["ee"], traj_results[traj_type]["target"]

        for i, (ee, tgt) in enumerate(zip(ee_list, target_list)):
            alpha = 0.4 + 0.3 * i / max(len(ee_list) - 1, 1)
            if i == 0:
                ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2],
                        "r--", linewidth=1.5, label="Desired", alpha=0.8)
                ax.plot(ee[:, 0],  ee[:, 1],  ee[:, 2],
                        "b-",  linewidth=1.0, label="Actual",  alpha=alpha)
            else:
                ax.plot(ee[:, 0], ee[:, 1], ee[:, 2],
                        "b-", linewidth=1.0, alpha=alpha)

        ax.set_title(titles[traj_type], fontsize=12)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Desired vs Actual End-Effector Trajectories", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_error_over_time(traj_results: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    titles = {"circle": "Circle", "figure8": "Figure-Eight", "lissajous": "Lissajous"}
    colours = {"1x": "steelblue", "2x": "darkorange", "3x": "firebrick"}

    for ax, traj_type in zip(axes, TRAJ_TYPES):
        for noise_label, errors_list in traj_results[traj_type]["noise_errors"].items():
            err_arr = np.array(errors_list)   # episodes × steps
            mean_err = err_arr.mean(axis=0)
            std_err  = err_arr.std(axis=0)
            steps = np.arange(len(mean_err))
            c = colours[noise_label]
            ax.plot(steps, mean_err * 100, color=c, label=f"σ {noise_label}", linewidth=1.5)
            ax.fill_between(steps,
                            (mean_err - std_err) * 100,
                            (mean_err + std_err) * 100,
                            alpha=0.15, color=c)
        ax.axhline(2.0, color="gray", linestyle=":", linewidth=1, label="2 cm threshold")
        ax.set_title(titles[traj_type])
        ax.set_xlabel("Step")
    axes[0].set_ylabel("Tracking error (cm)")
    axes[0].legend(fontsize=8)

    fig.suptitle("Tracking Error Over Episode (mean ± std)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_video(frames: list, path: str, fps: int = 25):
    if not frames:
        print(f"  [skip] No frames captured for {path}")
        return
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264",
                quality=8, macro_block_size=1)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs("results", exist_ok=True)

    model_path  = args.model
    vnorm_path  = args.vecnormalize

    print(f"\nLoading model from: {model_path}")
    print(f"VecNormalize from:  {vnorm_path}\n")

    traj_results = {}
    metrics_rows = []

    for traj_type in TRAJ_TYPES:
        print(f"--- {traj_type} ---")
        traj_results[traj_type] = {"ee": [], "target": [], "noise_errors": {}}

        for noise_label, sigma in [("1x", 0.01), ("2x", 0.02), ("3x", 0.03)]:
            model, vec_env = load_model(model_path, vnorm_path, traj_type, sigma)
            ee_list, tgt_list, err_list, act_list, frames = rollout(
                model, vec_env,
                n_episodes=3,
                capture_video=(noise_label == "1x"),
            )

            if noise_label == "1x":
                traj_results[traj_type]["ee"]     = ee_list
                traj_results[traj_type]["target"] = tgt_list

                # Save video
                video_path = f"results/rollout_{traj_type}.mp4"
                save_video(frames, video_path)

            traj_results[traj_type]["noise_errors"][noise_label] = err_list

            m = compute_metrics(err_list, act_list)
            metrics_rows.append({
                "trajectory": traj_type,
                "noise":      noise_label,
                **m,
            })
            print(f"  noise={noise_label}  mean={m['mean_error_m']*100:.1f}cm  "
                  f"max={m['max_error_m']*100:.1f}cm  "
                  f"RMSE={m['rmse_m']*100:.1f}cm  "
                  f"success={m['success_rate']*100:.0f}%  "
                  f"jerk={m['jerk_mean']:.4f}")

            vec_env.close()

    # --- Plots ---------------------------------------------------------------
    print("\nGenerating plots...")
    plot_3d_trajectories(traj_results, "results/trajectory_3d.png")
    plot_error_over_time(traj_results, "results/error_over_time.png")

    # --- Metrics table -------------------------------------------------------
    table_lines = [
        f"{'Trajectory':<12} {'Noise':>5} {'Mean err':>9} {'Max err':>9} "
        f"{'RMSE':>9} {'Success':>8} {'Jerk':>9}",
        "-" * 65,
    ]
    for r in metrics_rows:
        table_lines.append(
            f"{r['trajectory']:<12} {r['noise']:>5} "
            f"{r['mean_error_m']*100:>8.2f}cm "
            f"{r['max_error_m']*100:>8.2f}cm "
            f"{r['rmse_m']*100:>8.2f}cm "
            f"{r['success_rate']*100:>7.0f}% "
            f"{r['jerk_mean']:>9.4f}"
        )
    table = "\n".join(table_lines)
    print("\n" + table)

    with open("results/metrics_table.txt", "w") as f:
        f.write(table + "\n")
    print("\nSaved: results/metrics_table.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="models/best_model",
                        help="Path to saved PPO model (no .zip extension)")
    parser.add_argument("--vecnormalize", default="models/vecnormalize.pkl",
                        help="Path to VecNormalize stats file")
    args = parser.parse_args()
    main(args)
