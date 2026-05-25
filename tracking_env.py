"""
Franka Panda 3D end-effector trajectory tracking environment.

Observation (32-dim):
  q (7) | qdot (7) | ee_pos (3) | target_now (3) |
  target_t+5 (3) | target_t+10 (3) | error (3) | traj_one_hot (3)

Action (7-dim):
  Delta joint positions, clipped to +-0.05 rad.

Trajectories (trajectory-conditioned policy):
  circle    - 3D circle with vertical bob
  figure8   - Lemniscate figure-eight in XY with Z oscillation
  lissajous - 2:3 Lissajous curve (used as held-out test)
"""

import os
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import gymnasium_robotics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRANKA_HOME_QPOS = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])

JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.4],
    [-2.8973, 2.8973],
    [-1.6573, 2.1127],
    [-2.8973, 2.8973],
])

TRAJ_TYPES = ["circle", "figure8", "lissajous"]
TRAJ_ONE_HOT = {t: np.eye(3)[i] for i, t in enumerate(TRAJ_TYPES)}

_MESH_DIR = os.path.join(
    os.path.dirname(gymnasium_robotics.__file__),
    "envs", "assets", "kitchen_franka", "franka_assets", "meshes"
).replace("\\", "/")

# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def _build_xml() -> str:
    """Generate a standalone MuJoCo XML for the Franka arm (no fingers, no kitchen)."""
    return f"""<mujoco model="franka_reach">
  <compiler angle="radian" meshdir="{_MESH_DIR}"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <size nuser_actuator="5"/>

  <asset>
    <texture name="texplane" type="2d" builtin="checker"
             rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512"/>
    <material name="MatGnd" reflectance="0.5" texture="texplane"
              texrepeat="1 1" texuniform="true"/>

    <mesh name="link0_col" file="collision/link0.stl"/>
    <mesh name="link1_col" file="collision/link1.stl"/>
    <mesh name="link2_col" file="collision/link2.stl"/>
    <mesh name="link3_col" file="collision/link3.stl"/>
    <mesh name="link4_col" file="collision/link4.stl"/>
    <mesh name="link5_col" file="collision/link5.stl"/>
    <mesh name="link6_col" file="collision/link6.stl"/>
    <mesh name="link7_col" file="collision/link7.stl"/>
    <mesh name="hand_col"  file="collision/hand.stl"/>
    <mesh name="link0_viz" file="visual/link0.stl"/>
    <mesh name="link1_viz" file="visual/link1.stl"/>
    <mesh name="link2_viz" file="visual/link2.stl"/>
    <mesh name="link3_viz" file="visual/link3.stl"/>
    <mesh name="link4_viz" file="visual/link4.stl"/>
    <mesh name="link5_viz" file="visual/link5.stl"/>
    <mesh name="link6_viz" file="visual/link6.stl"/>
    <mesh name="link7_viz" file="visual/link7.stl"/>
    <mesh name="hand_viz"  file="visual/hand.stl"/>
  </asset>

  <default>
    <default class="panda">
      <joint pos="0 0 0" axis="0 0 1" limited="true"/>
      <position forcelimited="true" ctrllimited="true" user="1002 40 2001 -0.005 0.005"/>
      <default class="panda_viz">
        <geom contype="0" conaffinity="0" group="0" type="mesh" rgba=".95 .99 .92 1" mass="0"/>
      </default>
      <default class="panda_col">
        <geom contype="1" conaffinity="1" group="3" type="mesh" rgba=".5 .6 .7 1"/>
      </default>
      <default class="panda_arm">
        <joint damping="100"/>
      </default>
      <default class="panda_forearm">
        <joint damping="10"/>
      </default>
    </default>
  </default>

  <worldbody>
    <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3"
           pos="1  1 3" dir="-1 -1 -3"/>
    <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3"
           pos="1 -1 3" dir="-1  1 -3"/>
    <light directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3"
           pos="-1 0 3" dir=" 1  0 -3"/>
    <geom name="ground" pos="0 0 0" size="5 5 10" material="MatGnd"
          type="plane" contype="1" conaffinity="1"/>

    <!-- Kinematically driven target marker -->
    <body name="target_body" mocap="true" pos="0.4 0 0.55">
      <geom type="sphere" size="0.025" rgba="1 0.2 0.2 0.75"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Franka Panda (7-DOF arm, no fingers) -->
    <body name="panda0_link0" childclass="panda">
      <geom class="panda_viz" mesh="link0_viz"/>
      <geom class="panda_col" mesh="link0_col" mass="2.91242"/>
      <body name="panda0_link1" pos="0 0 0.333">
        <joint name="robot:panda0_joint1" range="-2.8973 2.8973" class="panda_arm"/>
        <geom class="panda_viz" mesh="link1_viz"/>
        <geom class="panda_col" mesh="link1_col" mass="2.7063"/>
        <body name="panda0_link2" pos="0 0 0" quat="0.707107 -0.707107 0 0">
          <joint name="robot:panda0_joint2" range="-1.7628 1.7628" class="panda_arm"/>
          <geom class="panda_viz" mesh="link2_viz"/>
          <geom class="panda_col" mesh="link2_col" mass="2.73046"/>
          <body name="panda0_link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0">
            <joint name="robot:panda0_joint3" range="-2.8973 2.8973" class="panda_arm"/>
            <geom class="panda_viz" mesh="link3_viz"/>
            <geom class="panda_col" mesh="link3_col" mass="2.04104"/>
            <body name="panda0_link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0">
              <joint name="robot:panda0_joint4" range="-3.0718 -0.4" class="panda_arm"/>
              <geom class="panda_viz" mesh="link4_viz"/>
              <geom class="panda_col" mesh="link4_col" mass="2.08129"/>
              <body name="panda0_link5" pos="-0.0825 0.384 0"
                    quat="0.707107 -0.707107 0 0">
                <joint name="robot:panda0_joint5" range="-2.8973 2.8973"
                       class="panda_forearm"/>
                <geom class="panda_viz" mesh="link5_viz"/>
                <geom class="panda_col" mesh="link5_col" mass="3.00049"/>
                <body name="panda0_link6" pos="0 0 0" euler="1.57 0 1.57">
                  <joint name="robot:panda0_joint6" range="-1.6573 2.1127"
                         class="panda_forearm"/>
                  <geom class="panda_viz" mesh="link6_viz"/>
                  <geom class="panda_col" mesh="link6_col" mass="1.3235"/>
                  <body name="panda0_link7" pos="0.088 0 0" euler="1.57 0 0.7854">
                    <joint name="robot:panda0_joint7" range="-2.8973 2.8973"
                           class="panda_forearm"/>
                    <geom class="panda_viz" mesh="link7_viz"/>
                    <geom class="panda_col" mesh="link7_col" mass="0.2"/>
                    <geom pos="0 0 0.107" quat="0.92388 0 0 -0.382683"
                          class="panda_viz" mesh="hand_viz"/>
                    <geom pos="0 0 0.107" quat="0.92388 0 0 -0.382683"
                          class="panda_col" mesh="hand_col" mass="0.81909"/>
                    <site name="end_effector" pos="0 0 0.210" size="0.01"
                          euler="0 0 -0.785398"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="panda0_joint1" joint="robot:panda0_joint1" class="panda"
              kp="870" forcerange="-87 87"  ctrlrange="-2.9671  2.9671"/>
    <position name="panda0_joint2" joint="robot:panda0_joint2" class="panda"
              kp="870" forcerange="-87 87"  ctrlrange="-1.8326  1.8326"/>
    <position name="panda0_joint3" joint="robot:panda0_joint3" class="panda"
              kp="870" forcerange="-87 87"  ctrlrange="-2.9671  2.9671"/>
    <position name="panda0_joint4" joint="robot:panda0_joint4" class="panda"
              kp="870" forcerange="-87 87"  ctrlrange="-3.1416  0.0"/>
    <position name="panda0_joint5" joint="robot:panda0_joint5" class="panda"
              kp="120" forcerange="-12 12"  ctrlrange="-2.9671  2.9671"/>
    <position name="panda0_joint6" joint="robot:panda0_joint6" class="panda"
              kp="120" forcerange="-12 12"  ctrlrange="-3.7525  2.1817"/>
    <position name="panda0_joint7" joint="robot:panda0_joint7" class="panda"
              kp="120" forcerange="-12 12"  ctrlrange="-2.9671  2.9671"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FrankaTrackingEnv(gym.Env):
    """
    End-effector trajectory tracking with the Franka Panda 7-DOF arm.

    The policy is conditioned on the trajectory type (circle / figure8 /
    lissajous) via a one-hot vector in the observation.  Training on all
    three simultaneously encourages generalisation across trajectory shapes.

    Observation noise models real-world sensor uncertainty:
      - joint positions: Gaussian sigma=0.01 rad
      - end-effector position: Gaussian sigma=0.02 m
    Rewards are computed on *true* (noiseless) values so the agent is
    incentivised to reduce actual tracking error, not just apparent error.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    # Trajectory parameters
    TRAJ_CENTER_OFFSET = np.array([0.0, 0.0, -0.15])
    TRAJ_RADIUS = 0.12       # metres
    TRAJ_PERIOD = 320        # steps for one full loop (at frame_skip=40 → 25.6 s) — slower = smoother
    TRAJ_Z_AMP = 0.05        # vertical bob amplitude

    # Noise sigmas (set to 0 to disable)
    NOISE_Q = 0.01
    NOISE_EE = 0.02

    # Action delta limit — smaller cap forces the policy to move gradually
    DELTA_CLIP = 0.04        # rad (was 0.05)

    # Reward weights — tuned for smooth motion
    R_TRACK  = 1.0    # tracking error²  (primary objective)
    R_ACTION = 0.05   # ||action||²      penalty on action magnitude   (was 0.01)
    R_SMOOTH = 0.5    # ||Δaction||²     jerk penalty — main smoothness lever (was 0.05)
    R_VEL    = 0.003  # ||qdot||²        joint-velocity penalty (new)
    R_BONUS  = 0.5    # Gaussian proximity bonus at ≲3 cm
    R_BONUS_K = 100.0 # sharpness of Gaussian

    def __init__(
        self,
        trajectory_type: str = "random",
        add_noise: bool = True,
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        frame_skip: int = 40,
    ):
        super().__init__()
        assert trajectory_type in TRAJ_TYPES + ["random"], \
            f"trajectory_type must be one of {TRAJ_TYPES + ['random']}"

        self.trajectory_type = trajectory_type
        self.add_noise = add_noise
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip

        # --- Build model from XML string written to a temp file ----------
        xml = _build_xml()
        self._xml_tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        )
        self._xml_tmpfile.write(xml)
        self._xml_tmpfile.flush()
        self._xml_tmpfile.close()

        self.model = mujoco.MjModel.from_xml_path(self._xml_tmpfile.name)
        self.data = mujoco.MjData(self.model)

        # --- Cache IDs ---------------------------------------------------
        self._joint_names = [f"robot:panda0_joint{i}" for i in range(1, 8)]
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self._joint_names
        ]
        self._qpos_ids = [self.model.jnt_qposadr[j] for j in self._joint_ids]
        self._qvel_ids = [self.model.jnt_dofadr[j]  for j in self._joint_ids]

        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body"
        )
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]

        # --- Compute trajectory centre from home FK ----------------------
        self.data.qpos[self._qpos_ids] = FRANKA_HOME_QPOS
        self.data.ctrl[:] = FRANKA_HOME_QPOS
        mujoco.mj_forward(self.model, self.data)
        self._traj_center = (
            self.data.site_xpos[self._ee_site_id].copy() + self.TRAJ_CENTER_OFFSET
        )

        # --- Spaces ------------------------------------------------------
        obs_dim = 7 + 7 + 3 + 3 + 3 + 3 + 3 + 3   # = 32
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.DELTA_CLIP,
            high=self.DELTA_CLIP,
            shape=(7,),
            dtype=np.float32,
        )

        # --- Episode state -----------------------------------------------
        self._step = 0
        self._prev_action = np.zeros(7)
        self._active_traj: str = "circle"

        # --- Renderer (lazy) --------------------------------------------
        self._renderer = None

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def _get_target(self, t: int, traj_type: str) -> np.ndarray:
        omega = 2 * np.pi / self.TRAJ_PERIOD
        c = self._traj_center
        R = self.TRAJ_RADIUS
        A = self.TRAJ_Z_AMP

        if traj_type == "circle":
            x = c[0] + R * np.cos(omega * t)
            y = c[1] + R * np.sin(omega * t)
            z = c[2] + A * np.sin(2 * omega * t)

        elif traj_type == "figure8":
            # Lemniscate-like: single loop in x, double-frequency in y
            x = c[0] + R * np.sin(omega * t)
            y = c[1] + R * np.sin(2 * omega * t) / 2
            z = c[2] + A * np.cos(2 * omega * t)

        else:  # lissajous (2:3 ratio, held-out)
            x = c[0] + R * np.sin(2 * omega * t + np.pi / 2)
            y = c[1] + R * np.sin(3 * omega * t)
            z = c[2] + A * np.sin(4 * omega * t)

        return np.array([x, y, z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Pick trajectory for this episode
        if self.trajectory_type == "random":
            self._active_traj = self.np_random.choice(TRAJ_TYPES)
        else:
            self._active_traj = self.trajectory_type

        mujoco.mj_resetData(self.model, self.data)

        # Home position + small Gaussian perturbation
        noise = self.np_random.normal(0, 0.05, size=7) if seed is None else \
            np.zeros(7)
        qpos = np.clip(
            FRANKA_HOME_QPOS + noise,
            JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
        )
        self.data.qpos[self._qpos_ids] = qpos
        self.data.ctrl[:] = qpos
        self.data.qvel[:] = 0.0

        self._step = 0
        self._prev_action = np.zeros(7)

        # Place target marker
        target = self._get_target(0, self._active_traj)
        self.data.mocap_pos[self._target_mocap_id] = target

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"trajectory_type": self._active_traj}
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.DELTA_CLIP, self.DELTA_CLIP)

        # Apply delta to position setpoint
        new_ctrl = self.data.ctrl[:7] + action
        new_ctrl = np.clip(new_ctrl, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        self.data.ctrl[:7] = new_ctrl

        # Simulate
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step += 1

        # True end-effector position (noiseless) for reward
        ee_true = self.data.site_xpos[self._ee_site_id].copy()

        # Advance trajectory
        target = self._get_target(self._step, self._active_traj)
        self.data.mocap_pos[self._target_mocap_id] = target
        mujoco.mj_forward(self.model, self.data)

        error = ee_true - target
        err_sq = float(np.dot(error, error))

        # Joint velocity (noiseless, used only for penalty)
        qd = self.data.qvel[self._qvel_ids]

        reward = (
            -self.R_TRACK  * err_sq
            -self.R_ACTION * float(np.dot(action, action))
            -self.R_SMOOTH * float(np.dot(action - self._prev_action,
                                          action - self._prev_action))
            -self.R_VEL    * float(np.dot(qd, qd))
            +self.R_BONUS  * float(np.exp(-self.R_BONUS_K * err_sq))
        )

        self._prev_action = action.copy()

        terminated = False
        truncated = self._step >= self.max_episode_steps
        obs = self._get_obs()
        info = {
            "trajectory_type": self._active_traj,
            "error_norm": float(np.linalg.norm(error)),
            "ee_pos": ee_true.copy(),
            "target_pos": target.copy(),
        }
        return obs.astype(np.float32), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        q  = self.data.qpos[self._qpos_ids].copy()
        qd = self.data.qvel[self._qvel_ids].copy()
        ee = self.data.site_xpos[self._ee_site_id].copy()

        if self.add_noise:
            q  = q  + self.np_random.normal(0, self.NOISE_Q,  size=7)
            ee = ee + self.np_random.normal(0, self.NOISE_EE, size=3)

        t = self._step
        ttype = self._active_traj
        target_now = self._get_target(t,      ttype)
        target_t5  = self._get_target(t +  5, ttype)
        target_t10 = self._get_target(t + 10, ttype)
        error = ee - target_now
        one_hot = TRAJ_ONE_HOT[ttype].copy()

        return np.concatenate([
            q, qd, ee, target_now, target_t5, target_t10, error, one_hot
        ]).astype(np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_xml_tmpfile"):
            try:
                os.unlink(self._xml_tmpfile.name)
            except OSError:
                pass
