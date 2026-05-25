"""
Smoke test: verify the environment loads, runs, and produces sane rewards.
Run this before starting the full training run.

    python smoke_test.py
"""

import numpy as np
from tracking_env import FrankaTrackingEnv, TRAJ_TYPES


def run_smoke_test():
    print("=" * 60)
    print("Franka Tracking Env — Smoke Test")
    print("=" * 60)

    for traj_type in TRAJ_TYPES:
        print(f"\nTrajectory: {traj_type}")
        env = FrankaTrackingEnv(trajectory_type=traj_type, add_noise=True)

        obs, info = env.reset(seed=42)
        assert obs.shape == (32,), f"Bad obs shape: {obs.shape}"
        assert not np.any(np.isnan(obs)), "NaN in first obs"
        print(f"  Obs shape : {obs.shape}   (no NaN)")
        print(f"  Traj type : {info['trajectory_type']}")
        print(f"  Traj center: {env._traj_center.round(3)}")

        rewards, errors = [], []
        for step in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert not np.any(np.isnan(obs)),    f"NaN in obs at step {step}"
            assert not np.isnan(reward),         f"NaN reward at step {step}"
            assert not np.isinf(reward),         f"Inf reward at step {step}"

            rewards.append(reward)
            errors.append(info["error_norm"])

            if terminated or truncated:
                break

        rewards = np.array(rewards)
        errors  = np.array(errors)

        print(f"  Steps run : {len(rewards)}")
        print(f"  Reward    : min={rewards.min():.4f}  max={rewards.max():.4f}"
              f"  mean={rewards.mean():.4f}")
        print(f"  EE error  : min={errors.min()*100:.1f}cm"
              f"  max={errors.max()*100:.1f}cm"
              f"  mean={errors.mean()*100:.1f}cm")

        assert rewards.min() > -5.0,  f"Reward too negative: {rewards.min():.4f}"
        assert errors.max()  < 2.0,   f"Error too large: {errors.max():.3f}m"
        print("  [PASS]")

        env.close()

    # Quick obs/action space sanity
    env = FrankaTrackingEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        a = env.action_space.sample()
        assert np.all(np.abs(a) <= 0.05 + 1e-6), "Action outside clip range"
    env.close()

    print("\n" + "=" * 60)
    print("All smoke tests passed. Safe to start training.")
    print("=" * 60)
    print("\nKick off training with:")
    print("  python train.py")
    print("\nOr a short test run:")
    print("  python train.py --timesteps 100000 --n_envs 4")


if __name__ == "__main__":
    run_smoke_test()
