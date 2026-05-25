"""
PPO training for Franka end-effector trajectory tracking.

Usage:
    python train.py                      # full 2M-step run
    python train.py --timesteps 500000   # shorter test run

Outputs:
    logs/ppo_tracking/          TensorBoard event files
    checkpoints/                model snapshots every 100k steps
    models/best_model.zip       best checkpoint (by mean reward)
    models/vecnormalize.pkl     running obs/reward statistics
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from tracking_env import FrankaTrackingEnv


def make_env(trajectory_type="random", add_noise=True, monitor=False):
    def _init():
        env = FrankaTrackingEnv(
            trajectory_type=trajectory_type,
            add_noise=add_noise,
        )
        if monitor:
            env = Monitor(env)
        return env
    return _init


def main(args):
    os.makedirs("logs/ppo_tracking", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # -----------------------------------------------------------------------
    # Vectorised training environment (8 parallel workers)
    # -----------------------------------------------------------------------
    n_envs = args.n_envs
    train_env = SubprocVecEnv([make_env("random", add_noise=True)] * n_envs,
                               start_method="spawn")
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # -----------------------------------------------------------------------
    # Evaluation environment (single env, fixed circle trajectory, no noise)
    # -----------------------------------------------------------------------
    eval_env = SubprocVecEnv([make_env("circle", add_noise=False, monitor=True)],
                              start_method="spawn")
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # raw rewards for eval
        training=False,      # don't update stats during eval
        clip_obs=10.0,
    )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path="checkpoints/",
        name_prefix="ppo_tracking",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/eval/",
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # -----------------------------------------------------------------------
    # PPO model
    # -----------------------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="logs/",
        verbose=1,
        seed=42,
    )

    print(f"\nTraining for {args.timesteps:,} timesteps on {n_envs} parallel envs.")
    print("TensorBoard: tensorboard --logdir logs/ppo_tracking\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ppo_tracking",
        progress_bar=True,
        reset_num_timesteps=True,
    )

    # -----------------------------------------------------------------------
    # Save final artefacts
    # -----------------------------------------------------------------------
    model.save("models/final_model")
    train_env.save("models/vecnormalize.pkl")
    print("\nSaved: models/final_model.zip  models/vecnormalize.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000,
        help="Total environment steps (default 2M)"
    )
    parser.add_argument(
        "--n_envs", type=int, default=8,
        help="Number of parallel environments (default 8)"
    )
    args = parser.parse_args()
    main(args)
