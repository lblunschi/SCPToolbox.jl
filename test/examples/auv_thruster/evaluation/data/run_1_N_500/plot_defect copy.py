from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from scipy.integrate import solve_ivp


@dataclass
class HydroParams:
    mass: float = 11.5
    inertia_z: float = 0.16
    added_mass_x: float = -5.5
    added_mass_y: float = -12.7
    added_mass_z: float = -14.57
    added_mass_yaw: float = -0.12
    linear_drag_x: float = -4.03
    linear_drag_y: float = -6.22
    linear_drag_z: float = -5.18
    linear_drag_yaw: float = -0.07
    quadratic_drag_x: float = -18.18
    quadratic_drag_y: float = -21.66
    quadratic_drag_z: float = -36.99
    quadratic_drag_yaw: float = -1.55
    buoyancy: float = 114.8
    weight: float = 112.8


@dataclass
class VehicleParams:
    thruster_allocation_matrix: np.ndarray


def abs_smooth(z: float, eps: float = 1e-7) -> float:
    return np.sqrt(z * z + eps * eps)


def get_current(
    state: np.ndarray,
    yawidx: int = 3,
    shear: float = 0.3,
    width: float = 0.5,
):
    x = state[0]
    psi = state[yawidx]

    # inertial shear in +/- y (smooth) WF
    v_i = -shear * np.tanh(x / width)

    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    # WF -> body
    u_c = spsi * v_i
    v_c = cpsi * v_i
    w_c = 0.0

    return u_c, v_c, w_c


def make_control_interpolator(t_u: np.ndarray, u_vals: np.ndarray):
    """
    Returns u(t) with linear interpolation over each control channel.
    u_vals shape: (6, M)
    """
    def u_of_t(t: float) -> np.ndarray:
        return np.array(
            [np.interp(t, t_u, u_vals[i, :]) for i in range(u_vals.shape[0])],
            dtype=float,
        )
    return u_of_t


def auv_dynamics(
    t: float,
    x: np.ndarray,
    u_of_t,
    hydro: HydroParams,
    veh: VehicleParams,
    tdil: float,
) -> np.ndarray:
    """
    State x:
      [x, y, z, yaw, u, v, w, r]
    """
    u_thr = u_of_t(t)  # shape (6,)
    TAM = veh.thruster_allocation_matrix

    pos = x[0:4]
    vel = x[4:8]

    psi = pos[3]
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # thrusters -> body force/moment [Fx, Fy, Fz, Mz]
    uxyzyaw = TAM @ u_thr

    # current
    u_c, v_c, w_c = get_current(pos)

    # helpers
    denx = hydro.mass - hydro.added_mass_x
    deny = hydro.mass - hydro.added_mass_y
    denz = hydro.mass - hydro.added_mass_z
    denr = hydro.inertia_z - hydro.added_mass_yaw

    urx = vel[0] - u_c
    ury = vel[1] - v_c
    urz = vel[2] - w_c

    yaw_moment = (
        (vel[1] - v_c) * (vel[0] - u_c) * (hydro.added_mass_y - hydro.added_mass_x)
        + (hydro.linear_drag_yaw + hydro.quadratic_drag_yaw * abs(vel[3])) * vel[3]
    )

    f = np.zeros(8, dtype=float)

    # kinematics
    f[0] = cpsi * vel[0] - spsi * vel[1]
    f[1] = spsi * vel[0] + cpsi * vel[1]
    f[2] = vel[2]
    f[3] = vel[3]

    # dynamics
    f[4] = (
        uxyzyaw[0]
        + (vel[1] - v_c) * (hydro.mass * vel[3] - hydro.added_mass_y * vel[3])
        + (hydro.linear_drag_x + hydro.quadratic_drag_x * abs(urx)) * urx
    ) / denx

    f[5] = (
        uxyzyaw[1]
        + (vel[0] - u_c) * (hydro.added_mass_x * vel[3] - hydro.mass * vel[3])
        + (hydro.linear_drag_y + hydro.quadratic_drag_y * abs(ury)) * ury
    ) / deny

    f[6] = (
        uxyzyaw[2]
        + hydro.buoyancy
        - hydro.weight
        + (hydro.linear_drag_z + hydro.quadratic_drag_z * abs(urz)) * urz
    ) / denz

    f[7] = (uxyzyaw[3] + yaw_moment) / denr

    # normalized-time scaling
    f *= tdil
    return f


def load_uc_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    t_u = df["t"].to_numpy(dtype=float)

    u_cols = [c for c in df.columns if c.startswith("u")]
    if len(u_cols) != 6:
        raise ValueError(f"Expected 6 control columns, got {u_cols}")

    u_vals = np.vstack([df[c].to_numpy(dtype=float) for c in u_cols])
    return t_u, u_vals


def load_xc_states(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads time and 8 state columns from a saved trajectory file.

    Expected columns:
      t, x1, ..., x8
    """
    df = pd.read_csv(csv_path)

    state_cols = [f"x{i}" for i in range(1, 9)]
    missing = [c for c in ["t", *state_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    t = df["t"].to_numpy(dtype=float)
    x = df[state_cols].to_numpy(dtype=float)  # shape (N, 8)
    return t, x


def integrate_one_step(
    t0: float,
    t1: float,
    x0: np.ndarray,
    u_of_t,
    hydro: HydroParams,
    veh: VehicleParams,
    tdil: float,
) -> np.ndarray:
    """
    Integrate one step from t0 to t1 starting at x0.
    Returns x(t1).
    """
    sol = solve_ivp(
        fun=lambda t, x: auv_dynamics(t, x, u_of_t, hydro, veh, tdil),
        t_span=(float(t0), float(t1)),
        y0=np.asarray(x0, dtype=float),
        t_eval=[float(t1)],
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed on [{t0}, {t1}]: {sol.message}")

    return sol.y[:, -1]


def compute_stepwise_defects(
    uc_csv_path: str | Path,
    xd_csv_path: str | Path,
    xd_ref_path: str | Path,
    tdil: float,
    out_integrated_csv: str | Path = "data_xc_integrated_stepwise.csv",
    out_defect_csv: str | Path = "data_defect.csv",
):
    """
    For each interval k -> k+1:
      - use saved state from xd_csv[k] as initial condition
      - integrate to t[k+1]
      - compare against xd_ref[k+1]
      - defect[k] = x_int[k+1] - x_ref[k+1]

    Assumes xd_csv and xd_ref use the same time grid.
    """
    hydro = HydroParams()

    TAM = np.array([
        [-0.707, -0.707,  0.707,  0.707, 0.0, 0.0],
        [ 0.707, -0.707,  0.707, -0.707, 0.0, 0.0],
        [ 0.0,    0.0,    0.0,    0.0,   1.0, 1.0],
        [ 0.1888, -0.1888, -0.1888, 0.1888, 0.0, 0.0],
    ], dtype=float)
    veh = VehicleParams(thruster_allocation_matrix=TAM)

    # controls
    t_u, u_vals = load_uc_csv(uc_csv_path)
    u_of_t = make_control_interpolator(t_u, u_vals)

    # discrete trajectory used as step initial conditions
    t_xd, x_xd = load_xc_states(xd_csv_path)

    # reference trajectory used to compute defects
    t_ref, x_ref = load_xc_states(xd_ref_path)

    if len(t_xd) != len(t_ref):
        raise ValueError("xd and reference trajectory do not have the same number of rows")

    if not np.allclose(t_xd, t_ref, atol=1e-12, rtol=0.0):
        raise ValueError("Time grids in xd and reference trajectory do not match")

    n = len(t_xd)
    if n < 2:
        raise ValueError("Need at least 2 time points to compute stepwise defects")

    integrated_rows = []
    integrated_rows.append({
        "k": 1,
        "t": t_xd[0],
        **{f"x{i}": x_xd[0, i - 1] for i in range(1, 9)},
    })

    defect_rows = []

    for k in range(n - 1):
        t0 = t_xd[k]
        t1 = t_xd[k + 1]

        x0_saved = x_xd[k, :]
        x1_ref = x_ref[k + 1, :]

        x1_int = integrate_one_step(
            t0=t0,
            t1=t1,
            x0=x0_saved,
            u_of_t=u_of_t,
            hydro=hydro,
            veh=veh,
            tdil=tdil,
        )

        defect = x1_int - x1_ref

        integrated_rows.append({
            "k": k + 2,
            "t": t1,
            **{f"x{i}": x1_int[i - 1] for i in range(1, 9)},
        })

        defect_rows.append({
            "k": k + 1,
            "t0": t0,
            "t1": t1,
            **{f"defect_{i}": defect[i - 1] for i in range(1, 9)},
            "defect_norm_2": float(np.linalg.norm(defect, ord=2)),
            "defect_norm_inf": float(np.linalg.norm(defect, ord=np.inf)),
        })

    df_integrated = pd.DataFrame(integrated_rows)
    df_defect = pd.DataFrame(defect_rows)

    df_integrated.to_csv(out_integrated_csv, index=False)
    df_defect.to_csv(out_defect_csv, index=False)

    return t_xd, x_xd, df_integrated, df_defect


def plot_trajectory(
    t_xd: np.ndarray,
    x_xd: np.ndarray,
    df_integrated: pd.DataFrame,
    out_xy_path: str | Path = "trajectory_xy.png",
    out_states_path: str | Path = "trajectory_states.png",
):
    """
    Plot:
      1) x-y trajectory
      2) all 8 states versus time
    xd  -> black circles
    integrated -> blue line / markers
    """
    t_int = df_integrated["t"].to_numpy(dtype=float)
    x_int = df_integrated[[f"x{i}" for i in range(1, 9)]].to_numpy(dtype=float)

    # --------------------------------------------------
    # 1) x-y trajectory
    # --------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.plot(x_xd[:, 0], x_xd[:, 1], "ko", markersize=5, label="xd")

    plt.plot(x_int[:, 0], x_int[:, 1], "b-", label="stepwise integrated")
    plt.plot(x_int[:, 0], x_int[:, 1], "bx", markersize=4)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_xy_path, dpi=200)
    plt.show()

    # --------------------------------------------------
    # 2) state trajectories vs time
    # --------------------------------------------------
    state_labels = ["x", "y", "z", "yaw", "u", "v", "w", "r"]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()

    for i in range(8):
        ax = axes[i]
        ax.plot(t_xd, x_xd[:, i], "ko", markersize=4, label="xd")

        ax.plot(t_int, x_int[:, i], "b-", label="stepwise integrated")
        ax.plot(t_int, x_int[:, i], "bx", markersize=3)
        ax.set_ylabel(state_labels[i])
        ax.grid(True)

        if i == 0:
            ax.legend()

    axes[-2].set_xlabel("t")
    axes[-1].set_xlabel("t")
    fig.suptitle("State trajectories", y=0.995)
    fig.tight_layout()
    fig.savefig(out_states_path, dpi=200)
    plt.show()


def plot_defects(
    df_defect: pd.DataFrame,
    out_defects_path: str | Path = "trajectory_defects.png",
):
    """
    Plot defect_i versus time for all 8 states, plus norm plots.
    """
    t_def = df_defect["t1"].to_numpy(dtype=float)
    defect = df_defect[[f"defect_{i}" for i in range(1, 9)]].to_numpy(dtype=float)

    state_labels = ["x", "y", "z", "yaw", "u", "v", "w", "r"]

    fig, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True)
    axes = axes.ravel()

    for i in range(8):
        ax = axes[i]
        ax.plot(t_def, defect[:, i], "r-", label=f"defect in {state_labels[i]}")
        ax.plot(t_def, defect[:, i], "ro", markersize=3)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_ylabel(state_labels[i])
        ax.grid(True)

        if i == 0:
            ax.legend()

    axes[8].plot(t_def, df_defect["defect_norm_2"].to_numpy(dtype=float), "m-")
    axes[8].plot(t_def, df_defect["defect_norm_2"].to_numpy(dtype=float), "mo", markersize=3)
    axes[8].set_ylabel("||def||_2")
    axes[8].grid(True)

    axes[9].plot(t_def, df_defect["defect_norm_inf"].to_numpy(dtype=float), "g-")
    axes[9].plot(t_def, df_defect["defect_norm_inf"].to_numpy(dtype=float), "go", markersize=3)
    axes[9].set_ylabel("||def||_inf")
    axes[9].grid(True)

    axes[-2].set_xlabel("t")
    axes[-1].set_xlabel("t")
    fig.suptitle("Stepwise defects", y=0.995)
    fig.tight_layout()
    fig.savefig(out_defects_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    base = Path(__file__).parent

    # load time-dilation parameter
    df_tdil = pd.read_csv(base / "data_p.csv")
    tdil = float(df_tdil.loc[0, "value"])

    t_xd, x_xd, df_integrated, df_defect = compute_stepwise_defects(
        uc_csv_path=base / "data_uc.csv",
        xd_csv_path=base / "data_xd.csv",
        xd_ref_path=base / "data_xd.csv",
        tdil=tdil,
        out_integrated_csv=base / "data_xc_integrated_stepwise.csv",
        out_defect_csv=base / "data_defect.csv",
    )

    plot_trajectory(
        t_xd=t_xd,
        x_xd=x_xd,
        df_integrated=df_integrated,
        out_xy_path=base / "trajectory_xy.png",
        out_states_path=base / "trajectory_states.png",
    )

    plot_defects(
        df_defect=df_defect,
        out_defects_path=base / "trajectory_defects.png",
    )

    print("Saved:")
    print(base / "data_xc_integrated_stepwise.csv")
    print(base / "data_defect.csv")
    print(base / "trajectory_xy.png")
    print(base / "trajectory_states.png")
    print(base / "trajectory_defects.png")