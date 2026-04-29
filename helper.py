"""Helper module for the SRBD-MPC quadruped tutorial.

Contains everything that students do NOT need to implement themselves:

  - install_acados / install_drake : notebook setup helpers
  - download_video / show_video    : small video utilities used by other tasks
  - RobotParams        : physical parameters of the "quadruped block"
  - GaitSequencer      : periodic contact schedule (trot, walk, pace, ...)
  - FootstepPlanner    : Raibert-style footstep planner for the MPC horizon
  - reference_trajectory(): simple constant-velocity reference generator
  - SRBDSimulator      : Drake-based floating-body simulator that receives
                         GRFs at known foot locations and returns the next state
  - plot_contact_schedule / plot_run / plot_feet_3d : visualisation helpers
  - run_controller_simulation / animate_swingup_simulation :
                         local double-pendulum simulation helpers
  - skew, Rz           : small math utilities

Leg index convention used throughout:
    0 = FR (Front  Right)
    1 = FL (Front  Left)
    2 = HR (Hind   Right)
    3 = HL (Hind   Left)

Body frame convention:
    x forward, y left, z up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _make_swingup_rhs_function(model):
    """Return a numerical RHS f(x, u) from an AcadosModel."""
    import casadi as ca

    return ca.Function(
        "double_pendulum_rhs",
        [model.x, model.u],
        [model.f_expl_expr],
    )


def _rk4_step(rhs, x, u, dt):
    """One explicit Runge-Kutta integration step."""
    k1 = np.array(rhs(x, u)).reshape(-1)
    k2 = np.array(rhs(x + 0.5 * dt * k1, u)).reshape(-1)
    k3 = np.array(rhs(x + 0.5 * dt * k2, u)).reshape(-1)
    k4 = np.array(rhs(x + dt * k3, u)).reshape(-1)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_controller_simulation(solver, model,
                              u_max_1, u_max_2, v_max,
                              Tf_sim=5.0,
                              sim_dt=0.002,
                              mpc_period=0.02,
                              x0=None,
                              warm_start_steps=200,
                              motor_response=1.0,
                              process_noise=None,
                              measurement_noise=None,
                              stop_on_solver_failure=False,
                              do_plot=True):
    """Run a swing-up NMPC controller against the notebook dynamics model.

    The helper mirrors the hardware ``run_controller`` interface, but replaces
    the CloudPendulum client with a local RK4 simulation. The controller is
    evaluated every ``mpc_period`` seconds and the torque is held constant
    between MPC updates.
    """
    rhs = _make_swingup_rhs_function(model)
    x = np.zeros(4) if x0 is None else np.array(x0, dtype=float).reshape(4)
    u_cmd = np.zeros(2)
    u_applied = np.zeros(2)

    process_noise = (np.zeros(4) if process_noise is None
                     else np.asarray(process_noise))
    measurement_noise = (np.zeros(4) if measurement_noise is None
                         else np.asarray(measurement_noise))

    # Warm-start the SQP trajectory from the simulation initial state.
    x_ws = x.copy()
    for _ in range(warm_start_steps):
        solver.set(0, "lbx", x_ws)
        solver.set(0, "ubx", x_ws)
        solver.solve()
        x_ws = np.array(solver.get(1, "x")).reshape(-1)

    ts, xs, us, solver_status = [], [], [], []
    next_mpc = 0.0
    t = 0.0
    while t < Tf_sim:
        if t + 1e-12 >= next_mpc:
            x_meas = x + np.random.normal(0.0, measurement_noise, size=4)
            x_meas[2:] = np.clip(x_meas[2:], -v_max, v_max)

            solver.set(0, "lbx", x_meas)
            solver.set(0, "ubx", x_meas)
            status = solver.solve()
            solver_status.append(status)

            if status != 0 and stop_on_solver_failure:
                print(f"Stopping simulation at t={t:.3f}s, solver status={status}")
                break

            candidate = np.array(solver.get(0, "u")).reshape(-1)
            if np.all(np.isfinite(candidate)):
                u_cmd = candidate
            u_cmd[0] = np.clip(u_cmd[0], -u_max_1, u_max_1)
            u_cmd[1] = np.clip(u_cmd[1], -u_max_2, u_max_2)
            next_mpc += mpc_period

        u_applied = u_applied + motor_response * (u_cmd - u_applied)
        x = _rk4_step(rhs, x, u_applied, sim_dt)
        if np.any(process_noise > 0.0):
            x += np.random.normal(0.0, process_noise, size=4)

        t += sim_dt
        ts.append(t)
        xs.append(x.copy())
        us.append(u_applied.copy())

    ts = np.asarray(ts)
    xs = np.asarray(xs)
    us = np.asarray(us)

    if len(ts) > 0 and do_plot:
        plot_run(ts, xs, us, solver_status=solver_status)
        err = np.array([
            np.arctan2(np.sin(xs[-1, 0] - np.pi), np.cos(xs[-1, 0] - np.pi)),
            np.arctan2(np.sin(xs[-1, 1]), np.cos(xs[-1, 1])),
        ])
        print("Final state:", xs[-1])
        print("Wrapped angle error to [pi, 0]:", err)
        if solver_status:
            unique, counts = np.unique(solver_status, return_counts=True)
            print("Solver status counts:", dict(zip(unique.tolist(),
                                                    counts.tolist())))

    return dict(ts=ts, xs=xs, us=us, solver_status=solver_status)


simulate_swingup_controller = run_controller_simulation


def animate_swingup_simulation(ts, xs, l1=0.05, l2=0.05, skip=5,
                               interval_ms=30):
    """Animate a simulated double-pendulum trajectory in a notebook."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation

    ts = np.asarray(ts)[::skip]
    xs = np.asarray(xs)[::skip]
    if len(ts) == 0:
        raise ValueError("No trajectory data to animate.")

    fig, ax = plt.subplots(figsize=(5, 5))
    reach = 1.1 * (l1 + l2)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], "-o", lw=4, markersize=8)
    title = ax.set_title("")

    def points(q1, q2):
        p0 = np.array([0.0, 0.0])
        p1 = p0 + np.array([l1 * np.sin(q1), -l1 * np.cos(q1)])
        p2 = p1 + np.array([l2 * np.sin(q1 + q2),
                            -l2 * np.cos(q1 + q2)])
        return np.vstack([p0, p1, p2])

    def update(i):
        p = points(xs[i, 0], xs[i, 1])
        line.set_data(p[:, 0], p[:, 1])
        title.set_text(f"t = {ts[i]:.2f} s")
        return line, title

    anim = FuncAnimation(fig, update, frames=len(ts), interval=interval_ms,
                         blit=True, repeat=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# ---------------------------------------------------------------------------
# General notebook setup / video helpers
# ---------------------------------------------------------------------------

def download_video(video_url):
    """Download a video and convert it to mp4."""
    import os
    import wget
    from pathlib import Path

    os.makedirs("videos", exist_ok=True)
    input_path = wget.download(video_url, out="videos/")
    output_path = Path(input_path).parent / (str(Path(input_path).stem) + ".mp4")
    convert_flv_to_mp4(input_path, output_path)
    return output_path


def convert_flv_to_mp4(input_path, output_path):
    """Convert an FLV file to MP4 using FFmpeg."""
    import subprocess

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "copy",
        "-c:a", "copy",
        output_path,
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        print(f"Conversion successful: {output_path}")
    else:
        print(f"Error during conversion: {process.stderr.decode()}")


def show_video(video_path):
    """Display an mp4 inside a Jupyter notebook."""
    from IPython.display import Video, display
    display(Video(str(video_path), embed=False))


def install_drake(update_interval=0.5):
    """Install pydrake and a few plotting/animation dependencies."""
    import sys
    import time
    import shlex
    import subprocess
    from IPython.display import clear_output

    def run(cmd, update_interval=update_interval):
        header = ">> " + " ".join(shlex.quote(str(c)) for c in cmd)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )

        last_line = ""
        last_update = 0.0

        clear_output(wait=True)
        print(header)

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            last_line = line
            now = time.time()

            if now - last_update >= update_interval:
                clear_output(wait=True)
                print(header)
                print(last_line)
                last_update = now

        ret = proc.wait()

        clear_output(wait=True)
        print(header)

        if ret != 0:
            print("FAILED:", last_line)
            raise subprocess.CalledProcessError(ret, cmd)

        print("DONE ✅")

    run([
        sys.executable, "-m", "pip", "install", "--break-system-packages",
        "drake", "numpy", "scipy", "matplotlib", "ipywidgets",
        "jupyter-server-proxy",
    ], update_interval=1.0)

    import pydrake
    clear_output(wait=True)
    print("Drake setup OK ✅")
    print("Drake path:", pydrake.getDrakePath())


def install_acados(force_build=True, update_interval=0.5):
    """Install/build acados and wire paths for the current notebook kernel."""
    import os
    import sys
    import time
    import shlex
    import subprocess
    from IPython.display import clear_output

    ACADOS_DIR = os.path.expanduser("~/acados")

    def run(cmd, cwd=None, update_interval=update_interval):
        header = ">> " + " ".join(shlex.quote(str(c)) for c in cmd)

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )

        last_line = ""
        last_update = 0.0

        clear_output(wait=True)
        print(header)

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            last_line = line
            now = time.time()

            if now - last_update >= update_interval:
                clear_output(wait=True)
                print(header)
                print(last_line)
                last_update = now

        ret = proc.wait()

        clear_output(wait=True)
        print(header)

        if ret != 0:
            print("FAILED:", last_line)
            raise subprocess.CalledProcessError(ret, cmd)

        print("DONE ✅")

    run([
        sys.executable, "-m", "pip", "install", "--break-system-packages",
        "casadi", "numpy", "scipy", "matplotlib", "cython", "deprecated",
    ])

    if not os.path.exists(ACADOS_DIR):
        run(["git", "clone", "https://github.com/acados/acados.git", ACADOS_DIR])
    else:
        clear_output(wait=True)
        print(f">> acados already exists at {ACADOS_DIR}")

    run(["git", "submodule", "update", "--recursive", "--init"], cwd=ACADOS_DIR)

    lib_path = os.path.join(ACADOS_DIR, "lib", "libacados.so")
    build_dir = os.path.join(ACADOS_DIR, "build")

    if force_build or not os.path.exists(lib_path):
        os.makedirs(build_dir, exist_ok=True)

        run(
            ["cmake", "-DACADOS_WITH_QPOASES=OFF", ".."],
            cwd=build_dir,
            update_interval=0.5,
        )

        run(
            ["make", "install", f"-j{os.cpu_count() or 2}"],
            cwd=build_dir,
            update_interval=1.0,
        )
    else:
        clear_output(wait=True)
        print(">> acados already built")

    os.environ["ACADOS_SOURCE_DIR"] = ACADOS_DIR

    lib_dir = os.path.join(ACADOS_DIR, "lib")
    old_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in old_ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{old_ld_path}"

    template_path = os.path.join(ACADOS_DIR, "interfaces", "acados_template")
    if template_path not in sys.path:
        sys.path.insert(0, template_path)

    import casadi  # noqa: F401
    from acados_template import AcadosOcp, AcadosOcpSolver  # noqa: F401

    clear_output(wait=True)
    print("Setup OK ✅")
    print("Python:", sys.executable)
    print("ACADOS_SOURCE_DIR:", os.environ["ACADOS_SOURCE_DIR"])


# ---------------------------------------------------------------------------
# Small math utilities
# ---------------------------------------------------------------------------

def skew(v):
    """Skew-symmetric matrix of a 3-vector such that skew(v) @ w == cross(v, w)."""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])


def Rz(yaw):
    """Rotation matrix about the world z-axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

@dataclass
class RobotParams:
    """Physical parameters of the quadruped-as-block model."""
    mass: float = 12.0                                          # kg
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.07, 0.26, 0.242]))  # body-frame I
    mu: float = 0.6                                             # friction coefficient
    f_min: float = 1.0                                          # min normal force per foot [N]
    f_max: float = 250.0                                        # max normal force per foot [N]
    body_size: tuple = (0.5, 0.25, 0.12)                        # Lx, Ly, Lz
    nominal_height: float = 0.30                                # nominal CoM height [m]
    hip_offsets: np.ndarray = field(default_factory=lambda: np.array([
        [ 0.22, -0.11, 0.0],   # FR
        [ 0.22,  0.11, 0.0],   # FL
        [-0.22, -0.11, 0.0],   # HR
        [-0.22,  0.11, 0.0],   # HL
    ]))
    g: float = 9.81

    @property
    def n_legs(self) -> int:
        return 4


LEG_NAMES = ("FR", "FL", "HR", "HL")


# ---------------------------------------------------------------------------
# Gait sequencer
# ---------------------------------------------------------------------------

class GaitSequencer:
    """Periodic gait with per-leg phase offsets and a common duty factor.

    contact = True  -> stance (foot on the ground, can push)
    contact = False -> swing  (foot in the air,       no force)

    Gait    period  offsets (FR,FL,HR,HL)   duty
    stand    0.5    (0, 0, 0, 0)            1.00
    trot     0.4    (0, .5, .5, 0)          0.5
    walk     0.8    (0, .5, .25, .75)       0.75
    pace     0.4    (0, .5, 0, .5)          0.5
    bound    0.4    (0, 0, .5, .5)          0.5
    """

    GAITS = {
        "stand": dict(period=0.5, offsets=np.array([0.0, 0.0, 0.0, 0.0]), duty=1.00),
        "trot":  dict(period=0.4, offsets=np.array([0.0, 0.5, 0.5, 0.0]), duty=0.50),
        "walk":  dict(period=0.8, offsets=np.array([0.0, 0.5, 0.25, 0.75]), duty=0.75),
        "pace":  dict(period=0.4, offsets=np.array([0.0, 0.5, 0.0, 0.5]), duty=0.50),
        "bound": dict(period=0.4, offsets=np.array([0.0, 0.0, 0.5, 0.5]), duty=0.50),
    }

    def __init__(self, gait: str = "trot"):
        if gait not in self.GAITS:
            raise ValueError(f"Unknown gait '{gait}'. Choose from {list(self.GAITS)}")
        self.name = gait
        g = self.GAITS[gait]
        self.period = float(g["period"])
        self.offsets = np.asarray(g["offsets"], dtype=float)
        self.duty = float(g["duty"])

    def phase(self, t: float) -> np.ndarray:
        """Per-leg phase in [0, 1)."""
        return ((t / self.period) + self.offsets) % 1.0

    def contact_state(self, t: float) -> np.ndarray:
        """(4,) boolean array: True = stance."""
        return self.phase(t) < self.duty

    def contact_schedule(self, t0: float, dt: float, N: int) -> np.ndarray:
        """Boolean schedule of shape (N+1, 4) over a horizon starting at t0."""
        return np.array([self.contact_state(t0 + k * dt) for k in range(N + 1)])

    def phase_within_stance(self, t: float) -> np.ndarray:
        """For each leg, the phase inside the current stance window, in [0, 1),
        or NaN if the leg is currently in swing."""
        ph = self.phase(t)
        out = np.where(ph < self.duty, ph / self.duty, np.nan)
        return out

    def phase_within_swing(self, t: float) -> np.ndarray:
        ph = self.phase(t)
        swing_ph = (ph - self.duty) / (1.0 - self.duty + 1e-12)
        return np.where(ph >= self.duty, swing_ph, np.nan)

    @property
    def stance_duration(self) -> float:
        return self.duty * self.period

    @property
    def swing_duration(self) -> float:
        return (1.0 - self.duty) * self.period


# ---------------------------------------------------------------------------
# Footstep planner (Raibert heuristic)
# ---------------------------------------------------------------------------

class FootstepPlanner:
    """Raibert-style footstep planner.

    For each leg and each MPC horizon step, returns a world-frame foot position.
    - For a leg currently in stance, the planted foothold is kept fixed during that
      stance window (as is standard for SRBD MPC).
    - For a leg in swing, we return the predicted next touchdown location so the
      MPC has a valid value to plug into B(r) -- though the force on a swing leg
      is constrained to zero anyway.

    Raibert heuristic:

        p_foot = p_hip_td  +  0.5 * v_body_td * T_stance  +  k * (v_body - v_cmd)

    with p_hip_td the hip position at touchdown, T_stance the stance duration of
    the gait, and k a small velocity-tracking feedback gain.
    """

    def __init__(self, params: RobotParams, gait: GaitSequencer, k_vel: float = 0.03):
        self.params = params
        self.gait = gait
        self.k_vel = float(k_vel)

    # ---- single-leg helpers ---------------------------------------------

    def _hip_world(self, p_body: np.ndarray, yaw: float, leg: int) -> np.ndarray:
        R = Rz(yaw)
        hip = p_body + R @ self.params.hip_offsets[leg]
        return hip

    def predicted_touchdown(self, p_body: np.ndarray, yaw: float,
                            v_body: np.ndarray, v_cmd: np.ndarray,
                            leg: int) -> np.ndarray:
        hip = self._hip_world(p_body, yaw, leg)
        foot = hip + 0.5 * np.asarray(v_body) * self.gait.stance_duration \
             + self.k_vel * (np.asarray(v_body) - np.asarray(v_cmd))
        foot[2] = 0.0
        return foot

    # ---- full horizon planning ------------------------------------------

    def plan_horizon(self,
                     current_feet: np.ndarray,
                     state_ref: np.ndarray,
                     v_cmd: np.ndarray,
                     t0: float, dt: float, N: int,
                     yaw_idx: int = 2, p_idx: slice = slice(3, 6),
                     v_idx: slice = slice(9, 12)) -> np.ndarray:
        """Plan (N+1, 4, 3) foot locations over the MPC horizon.

        current_feet : (4, 3) foot positions right now (where each foot currently is).
        state_ref    : (N+1, 12) reference trajectory; yaw at index `yaw_idx`,
                       CoM position at slice `p_idx`, linear velocity at `v_idx`.
        """
        feet = np.zeros((N + 1, 4, 3))
        planted = np.asarray(current_feet, dtype=float).copy()
        last_contact = self.gait.contact_state(t0)

        for k in range(N + 1):
            t_k = t0 + k * dt
            contact_k = self.gait.contact_state(t_k)

            yaw_k = float(state_ref[k, yaw_idx])
            p_k = np.asarray(state_ref[k, p_idx])
            v_k = np.asarray(state_ref[k, v_idx])

            for i in range(4):
                if contact_k[i] and not last_contact[i]:
                    # new touchdown at this step -> pick a foothold
                    planted[i] = self.predicted_touchdown(p_k, yaw_k, v_k, v_cmd, i)

                if contact_k[i]:
                    feet[k, i] = planted[i]
                else:
                    # leg is in swing: dummy value (force will be zero anyway)
                    feet[k, i] = self.predicted_touchdown(p_k, yaw_k, v_k, v_cmd, i)

            last_contact = contact_k
        return feet


# ---------------------------------------------------------------------------
# Reference trajectory
# ---------------------------------------------------------------------------

def reference_trajectory(p0: np.ndarray, yaw0: float,
                         v_cmd: np.ndarray, omega_cmd: float,
                         z_ref: float,
                         t0: float, dt: float, N: int) -> np.ndarray:
    """Constant-velocity reference trajectory of shape (N+1, 12).

    State layout (matches the MPC):
        x = [roll, pitch, yaw,   px, py, pz,   wx, wy, wz,   vx, vy, vz]

    ``p0`` and ``yaw0`` are the current measured pose. The horizon is therefore
    propagated with local time ``tau = k * dt``; ``t0`` is kept in the signature
    so callers can use the same argument set as contact/footstep planning.
    """
    ref = np.zeros((N + 1, 12))
    for k in range(N + 1):
        tau = k * dt
        yaw_k = yaw0 + omega_cmd * tau
        # integrate v_cmd in the commanded-yaw frame
        p_k = p0 + Rz(yaw0) @ (v_cmd * tau)
        p_k[2] = z_ref
        ref[k, 0] = 0.0                 # roll
        ref[k, 1] = 0.0                 # pitch
        ref[k, 2] = yaw_k               # yaw
        ref[k, 3:6] = p_k               # CoM position
        ref[k, 6] = 0.0                 # ang vel x
        ref[k, 7] = 0.0                 # ang vel y
        ref[k, 8] = omega_cmd           # ang vel z (yaw rate)
        ref[k, 9:12] = Rz(yaw_k) @ v_cmd  # linear velocity in world
    return ref


# ---------------------------------------------------------------------------
# Meshcat URL plumbing (JupyterHub / Colab / Deepnote / ...)
# ---------------------------------------------------------------------------

def start_meshcat(host: Optional[str] = None,
                  port: Optional[int] = None,
                  web_url_pattern: Optional[str] = None,
                  hub_host: Optional[str] = None,
                  verbose: bool = False):
    """Start a Meshcat server with sensible defaults for hosted Jupyters.

    Drake's :func:`StartMeshcat` does **not** recognise JupyterHub on its
    own, so this helper inspects the environment and patches
    ``MeshcatParams.web_url_pattern`` accordingly:

    - **JupyterHub**: if ``JUPYTERHUB_SERVICE_PREFIX`` is set (and you pass
      no explicit pattern), the URL becomes
      ``{hub_host}{JUPYTERHUB_SERVICE_PREFIX}proxy/{port}/``.
      ``hub_host`` defaults to an empty string (origin-relative), which
      works whenever you click the printed link from inside the hub.
      Requires ``jupyter-server-proxy`` to be installed **on the Jupyter
      server** (not just in the kernel!); a plain kernel restart after
      installing it is not enough — you must stop and start your hub
      server.
    - **Deepnote / Colab**: delegated to Drake's :func:`StartMeshcat`.
    - **Anywhere else**: vanilla ``http://{host}:{port}`` (localhost).

    Manual overrides::

        mc = start_meshcat(
            web_url_pattern="https://jupyter.dfki.de/user/myname/proxy/{port}/")
        sim = SRBDSimulator(params, meshcat=mc)
    """
    import os
    from pydrake.geometry import Meshcat, MeshcatParams, StartMeshcat

    hub_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")

    # auto-build a JupyterHub pattern when nothing is forced from outside
    if (web_url_pattern is None and host is None and port is None
            and hub_prefix is not None):
        web_url_pattern = (hub_host or "") + hub_prefix + "proxy/{port}/"
        if verbose:
            print(f"[meshcat] JupyterHub prefix: {hub_prefix}")
            print(f"[meshcat] proxy URL pattern: {web_url_pattern}")

    # no JupyterHub & no manual override → let Drake pick (handles Deepnote)
    if web_url_pattern is None and host is None and port is None:
        mc = StartMeshcat()
        if verbose:
            print(f"[meshcat] open this URL in your browser: {mc.web_url()}")
        return mc

    p = MeshcatParams()
    if host is not None:
        p.host = host
    if port is not None:
        p.port = int(port)
    if web_url_pattern is not None:
        p.web_url_pattern = web_url_pattern
    mc = Meshcat(p)
    if verbose:
        print(f"[meshcat] open this URL in your browser: {mc.web_url()}")
    return mc


# ---------------------------------------------------------------------------
# Drake simulator
# ---------------------------------------------------------------------------

class SRBDSimulator:
    """Minimal Drake-based simulator for the "quadruped-as-block".

    The block is a single free-floating rigid body with user-specified mass and
    inertia. At every step, the caller provides world-frame ground reaction
    forces f_i and world-frame application points p_foot_i.

    Each force is applied as an ExternallyAppliedSpatialForce at the
    corresponding foot location; Drake handles gravity and integration.
    """

    def __init__(self, params: RobotParams, dt: float = 0.002,
                 use_meshcat: bool = True,
                 ground_visual: bool = True,
                 force_visual: bool = True,
                 force_scale: float = 0.002,
                 meshcat=None):
        """
        Parameters
        ----------
        force_visual : bool
            If Meshcat is enabled, draw a live force vector for each foot.
        force_scale : float
            Meshcat force-vector scale in metres per Newton.
        meshcat : pydrake.geometry.Meshcat, optional
            Pass an existing Meshcat instance (e.g. created via
            ``StartMeshcat()`` or manually with custom ``MeshcatParams``).
            If ``None`` and ``use_meshcat`` is ``True``, a Meshcat will be
            created via ``StartMeshcat()``, which honours JupyterHub and
            Google Colab via ``jupyter-server-proxy`` so the browser URL
            matches the hub (not ``localhost``).
        """
        self.params = params
        self.dt = float(dt)
        self.use_meshcat = bool(use_meshcat)
        self.ground_visual = bool(ground_visual)
        self.force_visual = bool(force_visual)
        self.force_scale = float(force_scale)
        self._meshcat_user = meshcat
        self._build()

    # ---------------------------------------------------------------

    def _build(self):
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.multibody.tree import SpatialInertia, UnitInertia
        from pydrake.geometry import Box, Sphere, HalfSpace, Rgba
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.systems.analysis import Simulator

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        self.plant = plant
        self.scene_graph = scene_graph

        # --- floating block ------------------------------------------------
        m = self.params.mass
        I = self.params.inertia
        G = UnitInertia(I[0, 0] / m, I[1, 1] / m, I[2, 2] / m,
                        I[0, 1] / m, I[0, 2] / m, I[1, 2] / m)
        spatial_inertia = SpatialInertia(mass=m, p_PScm_E=np.zeros(3), G_SP_E=G)
        self.body = plant.AddRigidBody("block", spatial_inertia)

        Lx, Ly, Lz = self.params.body_size
        plant.RegisterVisualGeometry(
            self.body, RigidTransform(), Box(Lx, Ly, Lz),
            "block_visual", np.array([0.2, 0.3, 0.8, 1.0]))

        # --- ground (visual only, no collision) ----------------------------
        if self.ground_visual:
            X_ground = RigidTransform(RotationMatrix(), [0, 0, -0.01])
            plant.RegisterVisualGeometry(
                plant.world_body(), X_ground, Box(4.0, 4.0, 0.02),
                "ground_visual", np.array([0.85, 0.85, 0.85, 1.0]))

        plant.Finalize()

        # --- visualization -------------------------------------------------
        self.meshcat = None
        if self.use_meshcat:
            from pydrake.visualization import AddDefaultVisualization
            if self._meshcat_user is not None:
                self.meshcat = self._meshcat_user
            else:
                # StartMeshcat() picks up JUPYTERHUB_SERVICE_PREFIX and
                # friends and returns a Meshcat with a correctly-proxied
                # web URL (requires the jupyter-server-proxy package on the
                # hub, which is installed by ``install_drake()``).
                self.meshcat = start_meshcat()
            AddDefaultVisualization(builder=builder, meshcat=self.meshcat)

            # foot markers and force endpoints, updated every step
            self._foot_colors = [Rgba(1, 0.2, 0.2, 1), Rgba(0.2, 0.2, 1.0, 1),
                                 Rgba(1, 0.6, 0.2, 1), Rgba(0.2, 0.8, 0.3, 1)]
            for i, c in enumerate(self._foot_colors):
                self.meshcat.SetObject(f"/feet/{LEG_NAMES[i]}", Sphere(0.025), c)
                if self.force_visual:
                    self.meshcat.SetObject(
                        f"/forces/{LEG_NAMES[i]}/tip", Sphere(0.018), c)

        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        # NOTE: Simulator.set_publish_every_time_step was removed in Drake 2026.
        # Publish behaviour is now configured on the LeafSystem (e.g. via
        # MeshcatVisualizerParams.publish_period when calling AddDefaultVisualization).
        self.simulator.Initialize()
        self.context = self.simulator.get_mutable_context()
        self.plant_context = plant.GetMyMutableContextFromRoot(self.context)

    def _set_meshcat_line(self, path: str, points: np.ndarray, color, width: float = 4.0):
        """Set a Meshcat line while tolerating small Drake API differences."""
        try:
            self.meshcat.SetLine(path, points, line_width=width, rgba=color)
        except TypeError:
            self.meshcat.SetLine(path, points, width, color)

    def _update_force_arrow(self, leg: int, foot: np.ndarray, force: np.ndarray):
        """Draw one force as a shaft plus four arrowhead strokes in Meshcat."""
        from pydrake.math import RigidTransform

        color = self._foot_colors[leg]
        tip = foot + self.force_scale * force
        arrow = tip - foot
        length = float(np.linalg.norm(arrow))

        self._set_meshcat_line(
            f"/forces/{LEG_NAMES[leg]}/shaft",
            np.column_stack([foot, tip]),
            color,
            width=4.0)
        self.meshcat.SetTransform(
            f"/forces/{LEG_NAMES[leg]}/tip",
            RigidTransform(tip))

        if length < 1e-9:
            head_points = [tip, tip, tip, tip]
        else:
            direction = arrow / length
            ref = np.array([0.0, 0.0, 1.0])
            if abs(float(direction @ ref)) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            side_a = np.cross(direction, ref)
            side_a /= np.linalg.norm(side_a)
            side_b = np.cross(direction, side_a)
            side_b /= np.linalg.norm(side_b)

            head_len = min(0.08, 0.35 * length)
            head_width = 0.45 * head_len
            head_back = tip - head_len * direction
            head_points = [
                head_back + head_width * side_a,
                head_back - head_width * side_a,
                head_back + head_width * side_b,
                head_back - head_width * side_b,
            ]

        for j, base in enumerate(head_points):
            self._set_meshcat_line(
                f"/forces/{LEG_NAMES[leg]}/head_{j}",
                np.column_stack([tip, base]),
                color,
                width=4.0)

    # ---------------------------------------------------------------
    # state handling
    # ---------------------------------------------------------------

    def reset(self, p=None, rpy=None, v=None, omega=None):
        from pydrake.math import RigidTransform, RollPitchYaw
        from pydrake.multibody.math import SpatialVelocity

        p = np.asarray([0.0, 0.0, self.params.nominal_height]) if p is None else np.asarray(p)
        rpy = np.zeros(3) if rpy is None else np.asarray(rpy)
        v = np.zeros(3) if v is None else np.asarray(v)
        omega = np.zeros(3) if omega is None else np.asarray(omega)

        X_WB = RigidTransform(RollPitchYaw(rpy), p)
        self.plant.SetFreeBodyPose(self.plant_context, self.body, X_WB)
        self.plant.SetFreeBodySpatialVelocity(
            self.plant_context, self.body, SpatialVelocity(w=omega, v=v))
        self.context.SetTime(0.0)
        if self.meshcat is not None:
            self.simulator.get_system().ForcedPublish(self.context)

    def get_state(self) -> dict:
        """Return a dictionary with the current block state in world frame."""
        from pydrake.math import RollPitchYaw
        X_WB = self.plant.EvalBodyPoseInWorld(self.plant_context, self.body)
        V_WB = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.body)
        p = X_WB.translation().copy()
        R = X_WB.rotation().matrix().copy()
        rpy = RollPitchYaw(X_WB.rotation()).vector().copy()
        v = V_WB.translational().copy()
        omega = V_WB.rotational().copy()
        return dict(p=p, rpy=rpy, R=R, v=v, omega=omega, t=self.context.get_time())

    def state_vector_12(self) -> np.ndarray:
        """Return the state in the exact layout expected by the MPC."""
        s = self.get_state()
        x = np.zeros(12)
        x[0:3] = s["rpy"]
        x[3:6] = s["p"]
        x[6:9] = s["omega"]
        x[9:12] = s["v"]
        return x

    def state_vector_13(self) -> np.ndarray:
        """Backward-compatible alias for the old tutorial name."""
        return self.state_vector_12()

    # ---------------------------------------------------------------
    # stepping
    # ---------------------------------------------------------------

    def step(self, forces_world: np.ndarray, foot_positions_world: np.ndarray,
             duration: Optional[float] = None):
        """Advance the simulator while applying the given foot forces.

        forces_world          : (4, 3) forces [N]
        foot_positions_world  : (4, 3) application points in world frame [m]
        duration              : how long [s] to integrate the plant with the
                                given GRFs held constant. Defaults to ``self.dt``.
                                Drake's continuous-time plant chooses its own
                                internal integration step, so ``duration`` is the
                                full "zero-order-hold" interval that the forces
                                are applied for (typically the MPC period).
        """
        from pydrake.multibody.plant import ExternallyAppliedSpatialForce
        from pydrake.multibody.math import SpatialForce
        from pydrake.math import RigidTransform

        forces_world = np.asarray(forces_world, dtype=float)
        foot_positions_world = np.asarray(foot_positions_world, dtype=float)

        X_WB = self.plant.EvalBodyPoseInWorld(self.plant_context, self.body)
        p_WB = X_WB.translation()
        R_WB = X_WB.rotation().matrix()

        ext_forces = []
        for i in range(4):
            f = forces_world[i]
            r_world = foot_positions_world[i] - p_WB
            # application point expressed in the body frame
            p_BoBq_B = R_WB.T @ r_world
            ef = ExternallyAppliedSpatialForce()
            ef.body_index = self.body.index()
            ef.p_BoBq_B = p_BoBq_B
            ef.F_Bq_W = SpatialForce(tau=np.zeros(3), f=f)
            ext_forces.append(ef)

        self.plant.get_applied_spatial_force_input_port().FixValue(
            self.plant_context, ext_forces)

        # update foot markers and force vectors in meshcat
        if self.meshcat is not None:
            for i in range(4):
                self.meshcat.SetTransform(
                    f"/feet/{LEG_NAMES[i]}",
                    RigidTransform(foot_positions_world[i]))
                if self.force_visual:
                    self._update_force_arrow(
                        i, foot_positions_world[i], forces_world[i])

        dt = self.dt if duration is None else float(duration)
        t_new = self.context.get_time() + dt
        self.simulator.AdvanceTo(t_new)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_contact_schedule(gait: GaitSequencer, t0: float, dt: float, N: int, ax=None):
    """Render the contact schedule over a horizon as a stripe plot."""
    import matplotlib.pyplot as plt

    sched = gait.contact_schedule(t0, dt, N)  # (N+1, 4)
    times = t0 + dt * np.arange(N + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
    for i in range(4):
        stripe = np.where(sched[:, i], 1, 0)
        ax.fill_between(times, i - 0.4, i + 0.4,
                        where=stripe.astype(bool),
                        step="post", alpha=0.7,
                        label=LEG_NAMES[i] if i == 0 else None)
    ax.set_yticks(range(4))
    ax.set_yticklabels(LEG_NAMES)
    ax.set_xlabel("time [s]")
    ax.set_title(f"Contact schedule ({gait.name}, duty={gait.duty}, T={gait.period}s)")
    ax.set_xlim(times[0], times[-1])
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_feet_3d(state_history, feet_history, forces_history=None,
                 every: int = 20):
    """Simple 3-D view of the CoM trajectory, plus foot positions & forces
    sampled along it. Uses matplotlib (no Drake)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xs = np.asarray(state_history)
    feet = np.asarray(feet_history)   # (T, 4, 3)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs[:, 3], xs[:, 4], xs[:, 5], "-", lw=2, label="CoM")

    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for i in range(4):
        ax.scatter(feet[::every, i, 0], feet[::every, i, 1], feet[::every, i, 2],
                   s=14, color=colors[i], label=LEG_NAMES[i])

    if forces_history is not None:
        F = np.asarray(forces_history)
        scale = 1e-3
        for k in range(0, len(F), every):
            for i in range(4):
                p = feet[k, i]
                f = F[k, i] * scale
                ax.plot([p[0], p[0] + f[0]],
                        [p[1], p[1] + f[1]],
                        [p[2], p[2] + f[2]], "-k", alpha=0.4)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend()
    ax.set_title("CoM trajectory and foot positions")
    return ax


def plot_footstep_plan_2d(ref, feet_horizon, contact_schedule=None, ax=None):
    """Top-down view of the discrete MPC horizon samples.

    ref            : (N+1, 12) reference states
    feet_horizon   : (N+1, 4, 3) absolute world foot positions
    contact_schedule : optional (N+1, 4) stance flags
    """
    import matplotlib.pyplot as plt

    ref = np.asarray(ref)
    feet = np.asarray(feet_horizon)
    if contact_schedule is not None:
        contact_schedule = np.asarray(contact_schedule, dtype=bool)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(ref[:, 3], ref[:, 4], marker="o", s=28, color="k",
               label="CoM samples")
    ax.scatter(ref[0, 3], ref[0, 4], marker="o", s=85, color="k",
               label="stage 0")
    ax.scatter(ref[-1, 3], ref[-1, 4], marker="x", s=80, color="k",
               label=f"stage {len(ref) - 1}")
    for k in range(0, len(ref), max(1, len(ref) // 5)):
        ax.text(ref[k, 3], ref[k, 4], f" {k}", color="k", fontsize=8,
                va="bottom")

    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, name in enumerate(LEG_NAMES):
        xy = feet[:, i, :2]

        if contact_schedule is None:
            ax.scatter(xy[:, 0], xy[:, 1], s=28, color=colors[i], label=name)
        else:
            stance = contact_schedule[:, i]
            ax.scatter(xy[stance, 0], xy[stance, 1], s=36,
                       color=colors[i], label=f"{name} stance")

        ax.text(xy[0, 0], xy[0, 1], f" {name}", color=colors[i],
                fontsize=9, va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title("Discrete MPC horizon samples and stance footholds")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    return ax


def plot_swingup_run(ts, xs, us, taus_meas=None, solver_status=None, q_target=None):
    """Plot the closed-loop swing-up run used by older exercises."""
    import matplotlib.pyplot as plt

    ts = np.asarray(ts)
    xs = np.asarray(xs)
    us = np.asarray(us)
    if q_target is None:
        q_target = np.array([np.pi, 0.0])

    # --- joint angles ---
    plt.figure(figsize=(10, 4))
    plt.plot(ts, xs[:, 0], label="q1")
    plt.plot(ts, xs[:, 1], label="q2")
    plt.axhline(q_target[0], linestyle="--", color="tab:blue", alpha=0.5)
    plt.axhline(q_target[1], linestyle="--", color="tab:orange", alpha=0.5)
    plt.xlabel("time [s]")
    plt.ylabel("angle [rad]")
    plt.title("Joint angles")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- joint velocities ---
    plt.figure(figsize=(10, 4))
    plt.plot(ts, xs[:, 2], label="dq1")
    plt.plot(ts, xs[:, 3], label="dq2")
    plt.xlabel("time [s]")
    plt.ylabel("velocity [rad/s]")
    plt.title("Joint velocities")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- commanded (and optionally measured) torques ---
    plt.figure(figsize=(10, 4))
    plt.plot(ts, us[:, 0], label="u1 cmd")
    plt.plot(ts, us[:, 1], label="u2 cmd")
    if taus_meas is not None:
        taus_meas = np.asarray(taus_meas)
        plt.plot(ts, taus_meas[:, 0], "--", label="tau1 meas", alpha=0.7)
        plt.plot(ts, taus_meas[:, 1], "--", label="tau2 meas", alpha=0.7)
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.title("Commanded / measured joint torques")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_run(ts, xs, us, contacts=None, ref=None,
             taus_meas=None, solver_status=None, q_target=None):
    """Overview plot of a closed-loop run.

    xs: (T, 12)  state history [rpy, p, omega, v]
    us: (T, 12)  force history (4 x 3, flattened per step)
    contacts: (T, 4) optional stance flags
    ref: (T, 12) optional reference trajectory for overlay

    For backwards compatibility with older swing-up exercises, this function
    dispatches to ``plot_swingup_run`` when ``xs`` has four states or when
    swing-up-specific arguments are provided.
    """
    import matplotlib.pyplot as plt

    ts = np.asarray(ts); xs = np.asarray(xs); us = np.asarray(us)
    if (xs.ndim == 2 and xs.shape[1] == 4) or taus_meas is not None \
            or solver_status is not None or q_target is not None:
        return plot_swingup_run(ts, xs, us, taus_meas=taus_meas,
                                solver_status=solver_status,
                                q_target=q_target)

    fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    # --- CoM position ---
    for j, lbl in enumerate(["x", "y", "z"]):
        axs[0].plot(ts, xs[:, 3 + j], label=f"p{lbl}")
        if ref is not None:
            axs[0].plot(ts, ref[:, 3 + j], "--", alpha=0.5, color=f"C{j}")
    axs[0].set_ylabel("CoM position [m]")
    axs[0].legend(); axs[0].grid(True)

    # --- CoM linear velocity ---
    for j, lbl in enumerate(["vx", "vy", "vz"]):
        axs[1].plot(ts, xs[:, 9 + j], label=lbl)
        if ref is not None:
            axs[1].plot(ts, ref[:, 9 + j], "--", alpha=0.5, color=f"C{j}")
    axs[1].set_ylabel("CoM linear velocity [m/s]")
    axs[1].legend(); axs[1].grid(True)

    # --- Orientation (rpy) ---
    for j, lbl in enumerate(["roll", "pitch", "yaw"]):
        axs[2].plot(ts, xs[:, j], label=lbl)
        if ref is not None:
            axs[2].plot(ts, ref[:, j], "--", alpha=0.5, color=f"C{j}")
    axs[2].set_ylabel("orientation [rad]")
    axs[2].legend(); axs[2].grid(True)

    # --- Forces (z-component per leg) ---
    for i in range(4):
        axs[3].plot(ts, us[:, 3 * i + 2], label=f"f_z {LEG_NAMES[i]}")
    if contacts is not None:
        c = np.asarray(contacts)
        for i in range(4):
            axs[3].fill_between(ts, -5, 0, where=c[:, i],
                                step="post", alpha=0.15, color=f"C{i}")
    axs[3].set_ylabel("vertical GRF [N]"); axs[3].set_xlabel("time [s]")
    axs[3].legend(ncol=4, fontsize=8); axs[3].grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Inline matplotlib animation (no Meshcat / no server required)
# ---------------------------------------------------------------------------

def _rpy_to_R(rpy):
    r, p, y = rpy
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rzz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rzz @ Ry @ Rx                            # ZYX Euler


def animate_run(xs, feet, forces=None,
                body_size=(0.5, 0.25, 0.12),
                dt: float = 0.03,
                stride: int = 1,
                fps: int = 30,
                force_scale: float = 2e-3,
                figsize=(9, 6),
                elev=22, azim=-55,
                use_video: bool = False):
    """Render an inline matplotlib 3-D animation of a closed-loop SRBD run.

    Works anywhere Python + matplotlib runs — no Meshcat, no proxy, no hub.

    Parameters
    ----------
    xs      : (T, 12) state log  (rpy, p, omega, v)
    feet    : (T, 4, 3) foot positions
    forces  : (T, 4, 3), optional — ground reaction forces (drawn as arrows)
    dt      : timestep between samples [s]
    stride  : subsample every Nth frame to keep the HTML small
    fps     : playback frame rate
    use_video : if True, encode as an H.264 HTML5 video (needs ffmpeg).
                Otherwise fall back to ``to_jshtml`` (no extra deps).

    Returns
    -------
    IPython ``HTML`` object that Jupyter renders inline.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from IPython.display import HTML

    xs = np.asarray(xs)
    feet = np.asarray(feet)
    T = xs.shape[0]
    frames = np.arange(0, T, max(1, int(stride)))

    Lx, Ly, Lz = body_size
    vb = 0.5 * np.array([
        [-Lx, -Ly, -Lz], [+Lx, -Ly, -Lz], [+Lx, +Ly, -Lz], [-Lx, +Ly, -Lz],
        [-Lx, -Ly, +Lz], [+Lx, -Ly, +Lz], [+Lx, +Ly, +Lz], [-Lx, +Ly, +Lz],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    pad = 0.3
    xmin = min(xs[:, 3].min(), feet[:, :, 0].min()) - pad
    xmax = max(xs[:, 3].max(), feet[:, :, 0].max()) + pad
    ymin = min(xs[:, 4].min(), feet[:, :, 1].min()) - pad
    ymax = max(xs[:, 4].max(), feet[:, :, 1].max()) + pad
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(0.0, 0.8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")

    trail, = ax.plot([], [], [], "-k", lw=0.8, alpha=0.4)
    body_lines = Line3DCollection([], colors="tab:blue", lw=2)
    # autolim=False avoids a matplotlib crash when the collection is empty
    ax.add_collection3d(body_lines, autolim=False)

    foot_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    foot_dots = [ax.plot([], [], [], "o", ms=8, color=c)[0] for c in foot_colors]
    force_lines = Line3DCollection([], colors="k", lw=1.2, alpha=0.7) if forces is not None else None
    if force_lines is not None:
        ax.add_collection3d(force_lines, autolim=False)
    if forces is not None:
        forces = np.asarray(forces)

    title = ax.set_title("")

    def update(k):
        p_com = xs[k, 3:6]
        R = _rpy_to_R(xs[k, 0:3])
        vw = (R @ vb.T).T + p_com
        body_lines.set_segments([[vw[a], vw[b]] for a, b in edges])

        trail.set_data(xs[:k + 1, 3], xs[:k + 1, 4])
        trail.set_3d_properties(xs[:k + 1, 5])

        for i in range(4):
            foot_dots[i].set_data([feet[k, i, 0]], [feet[k, i, 1]])
            foot_dots[i].set_3d_properties([feet[k, i, 2]])

        if force_lines is not None:
            segs = []
            for i in range(4):
                a = feet[k, i]
                b = a + forces[k, i] * force_scale
                segs.append([a, b])
            force_lines.set_segments(segs)

        title.set_text(f"t = {k * dt:.2f} s")
        return [body_lines, trail, title, *foot_dots] + \
               ([force_lines] if force_lines is not None else [])

    ani = FuncAnimation(fig, update, frames=frames,
                        interval=1000.0 / fps, blit=False)
    plt.close(fig)
    if use_video:
        try:
            return HTML(ani.to_html5_video())
        except Exception as e:
            print(f"[animate_run] ffmpeg unavailable ({e!s}); using jshtml.")
    return HTML(ani.to_jshtml(fps=fps))


# ---------------------------------------------------------------------------
# Initial foot placement helpers
# ---------------------------------------------------------------------------

def initial_feet(params: RobotParams, p_body: np.ndarray, yaw: float) -> np.ndarray:
    """Place all 4 feet directly under their hips on the ground."""
    feet = np.zeros((4, 3))
    R = Rz(yaw)
    for i in range(4):
        hip_w = p_body + R @ params.hip_offsets[i]
        feet[i] = np.array([hip_w[0], hip_w[1], 0.0])
    return feet
