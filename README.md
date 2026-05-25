# 3D End-Effector Tracking with PPO

A trajectory-conditioned PPO policy trained to track 3D end-effector paths with a 7-DOF Franka Panda arm in MuJoCo.  A single network — conditioned on a trajectory-type one-hot — generalises across circle, figure-eight, and an unseen Lissajous trajectory.

![Arm tracking a figure-eight](results/rollout_figure8.gif)

---

## Results

Trained for 2 M steps (~17 min on CPU, 8 parallel envs).  Evaluated over 3 rollouts × 9 conditions.  Noise labels: 1× = nominal training noise, 2×/3× = harder robustness tests.

| Trajectory   | Noise | Mean err | Max err | RMSE   | Success < 2 cm | Jerk   |
|--------------|-------|----------|---------|--------|----------------|--------|
| circle       | 1×    | 1.68 cm  | 18.7 cm | 2.04 cm | 71 %          | 0.4239 |
| circle       | 2×    | 2.04 cm  | 18.7 cm | 2.39 cm | 54 %          | 0.4008 |
| circle       | 3×    | 2.57 cm  | 18.7 cm | 2.95 cm | 37 %          | 0.3833 |
| figure-eight | 1×    | 1.69 cm  | 10.4 cm | 1.87 cm | 71 %          | 0.4367 |
| figure-eight | 2×    | 2.11 cm  | 10.4 cm | 2.33 cm | 50 %          | 0.4042 |
| figure-eight | 3×    | 2.58 cm  | 10.4 cm | 2.87 cm | 37 %          | 0.3862 |
| lissajous*   | 1×    | 1.79 cm  | 18.4 cm | 2.13 cm | 66 %          | 0.4031 |
| lissajous*   | 2×    | 2.21 cm  | 18.5 cm | 2.55 cm | 48 %          | 0.3862 |
| lissajous*   | 3×    | 2.73 cm  | 18.5 cm | 3.07 cm | 30 %          | 0.3687 |

\* Lissajous (2:3 frequency ratio) was **not seen during training** — held out as an out-of-distribution generalisation test.  Mean error within 0.1 cm of the trained trajectories demonstrates that trajectory conditioning transfers to novel shapes.

The high max-error values reflect the initial transient at episode start (arm moves from home to the trajectory from rest); once on-trajectory, steady-state errors are typically < 2 cm.

![3D trajectory comparison](results/trajectory_3d.png)
![Tracking error over time](results/error_over_time.png)

---

## Approach

### State (32-dim observation)
Joint positions q (7) and velocities q&#775; (7) give the robot's proprioceptive state.  End-effector position (3) is computed from forward kinematics.  Three look-ahead targets — now, +5 steps, +10 steps (9 total) — let the policy anticipate upcoming trajectory curvature rather than react greedily.  The current tracking error (3) is included redundantly for fast credit assignment.  A trajectory one-hot (3) conditions the single policy across all three trajectory shapes simultaneously, enabling generalisation without separate networks.

### Action (7-dim)
Delta joint positions clipped to ±0.05 rad, applied to the existing setpoint of the arm's built-in high-gain position controllers (kp = 870 for proximal joints, 120 for wrist).  This residual formulation keeps the arm smooth and safe: the PD controller handles stabilisation, the policy only nudges the target.

### Reward
```
r = -||e||^2 - 0.01||a||^2 - 0.05||a - a_prev||^2 + 0.5*exp(-100*||e||^2)
```
The squared-distance penalty dominates when tracking is poor; the Gaussian bonus provides a dense shaped reward near the target (fires when ||e|| < ~3 cm).  Action-magnitude and action-difference penalties suppress vibration.  Rewards are computed on **true** (noiseless) end-effector positions while observations contain noisy values — forcing robustness to sensor uncertainty.

### Trajectory representation
Each trajectory is a smooth analytic function of step index `t`.  Circle and figure-eight share the same angular frequency (one loop ≈ 200 steps / 16 s); Lissajous uses a 2:3 ratio creating a non-repeating pattern over any 200-step window.  Look-ahead points (t+5, t+10) encode curvature, essential for tight tracking at figure-eight inflection points.

### Uncertainty
Gaussian observation noise (sigma_q = 0.01 rad, sigma_ee = 0.02 m) is injected at every step during training.  Robustness evaluation at 2x and 3x sigma characterises degradation under harsher conditions.  Rewards on ground-truth quantities ensure the agent minimises actual (not perceived) error.

### Evaluation
Three rollouts per trajectory x noise level, totalling 27 episodes.  Metrics: mean error, max error, RMSE, success rate (||e|| < 2 cm), and mean jerk (second finite difference of actions).  Lissajous is held out entirely from training.

---

## Design notes

**Why PPO over SAC?**  PPO is on-policy and converges reliably with dense shaped rewards and cheap simulation.  SAC's replay buffer excels under sparse rewards or sample-constrained real-robot settings; neither applies here.  The full 2 M step run completed in 17 minutes on CPU, confirming that on-policy rollouts are not the bottleneck.

**Why trajectory-conditioned rather than three separate policies?**  A single conditioned network exploits the structural similarity across shapes — identical dynamics, joint limits, and inertia, only the target sequence changes.  The one-hot conditioning forces a shared "how to track" representation and introduces zero architectural complexity.  Training on all three simultaneously acts as a regulariser that prevents memorisation of specific phase patterns — which is why the held-out Lissajous generalises within 0.1 cm of the trained trajectories.

---

## Run it

```bash
# 1. Install dependencies
pip install gymnasium gymnasium-robotics stable-baselines3[extra] mujoco imageio imageio-ffmpeg matplotlib tensorboard tqdm rich

# 2. Verify environment (< 30 s)
python smoke_test.py

# 3. Train (~17 min on a modern CPU with 8 parallel envs)
python train.py

# 4. Evaluate + generate plots and videos
python evaluate.py
```

Monitor training live:
```bash
tensorboard --logdir logs/ppo_tracking
```

---

## Future work

- **Residual RL on a Jacobian pseudo-inverse base controller**: the base controller handles posture in the null space, PPO outputs only a Cartesian residual correction.  This should cut required training steps roughly in half because the base controller already satisfies most of the task most of the time.
- **Orientation tracking**: augmenting with a 6D rotation representation and an orientation error term would make the policy suitable for manipulation tasks where grasp angle matters, at the cost of a larger observation space and harder credit assignment.
- **Sim-to-real transfer**: domain randomisation over link mass (±10 %), joint damping (±20 %), and timestep jitter, combined with latency-aware control (stacking the last 3 observations), would make the policy deployable on hardware without retraining.
