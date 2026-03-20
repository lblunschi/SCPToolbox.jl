from __future__ import annotations

import numpy as np
import pandas as pd
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
    shear: float = 0.2,
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
    u_thr = u_of_t(t)
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
        (vel[1] - v_c) * (vel[0] - u_c) * (hydro.added_mass_y - hydro.added_mass_x)*0.8
        + (hydro.linear_drag_yaw + hydro.quadratic_drag_yaw * abs_smooth(vel[3])) * vel[3]
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
        + (hydro.linear_drag_x + hydro.quadratic_drag_x * abs_smooth(urx)) * urx
    ) / denx

    f[5] = (
        uxyzyaw[1]
        + (vel[0] - u_c) * (hydro.added_mass_x * vel[3] - hydro.mass * vel[3])
        + (hydro.linear_drag_y + hydro.quadratic_drag_y * abs_smooth(ury)) * ury
    ) / deny

    f[6] = (
        uxyzyaw[2]
        + hydro.buoyancy
        - hydro.weight
        + (hydro.linear_drag_z + hydro.quadratic_drag_z * abs_smooth(urz)) * urz
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


def compute_iterative_rollout_defects(
    uc_csv_path: str | Path,
    xc_ref_path: str | Path,
    tdil: float,
    out_rollout_csv: str | Path = "data_xc_iterative_rollout.csv",
    out_defect_csv: str | Path = "data_defect_iterative.csv",
):
    """
    Chained rollout:
      x_roll[0] = x_ref[0]
      x_roll[k+1] = Phi(x_roll[k], u, t_k -> t_{k+1})

    Defect at node k:
      defect[k] = x_roll[k] - x_ref[k]

    This shows how error accumulates over time.
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

    # reference trajectory
    t_ref, x_ref = load_xc_states(xc_ref_path)

    n = len(t_ref)
    if n < 2:
        raise ValueError("Need at least 2 time points to compute iterative defects")

    # rollout storage
    x_roll = np.zeros_like(x_ref)
    x_roll[0, :] = x_ref[0, :]

    rollout_rows = []
    defect_rows = []

    # initial node
    defect0 = x_roll[0, :] - x_ref[0, :]
    rollout_rows.append({
        "k": 1,
        "t": t_ref[0],
        **{f"x_roll_{i}": x_roll[0, i - 1] for i in range(1, 9)},
        **{f"x_ref_{i}": x_ref[0, i - 1] for i in range(1, 9)},
    })
    defect_rows.append({
        "k": 1,
        "t": t_ref[0],
        **{f"defect_{i}": defect0[i - 1] for i in range(1, 9)},
        "defect_norm_2": float(np.linalg.norm(defect0, ord=2)),
        "defect_norm_inf": float(np.linalg.norm(defect0, ord=np.inf)),
    })

    # chained propagation
    for k in range(n - 1):
        t0 = t_ref[k]
        t1 = t_ref[k + 1]

        x0_roll = x_roll[k, :]
        x1_roll = integrate_one_step(
            t0=t0,
            t1=t1,
            x0=x0_roll,
            u_of_t=u_of_t,
            hydro=hydro,
            veh=veh,
            tdil=tdil,
        )

        x_roll[k + 1, :] = x1_roll
        

        defect = x_roll[k + 1, :] - x_ref[k + 1, :]

        rollout_rows.append({
            "k": k + 2,
            "t": t1,
            **{f"x_roll_{i}": x_roll[k + 1, i - 1] for i in range(1, 9)},
            **{f"x_ref_{i}": x_ref[k + 1, i - 1] for i in range(1, 9)},
        })

        defect_rows.append({
            "k": k + 2,
            "t": t1,
            "t0": t0,
            "t1": t1,
            **{f"defect_{i}": defect[i - 1] for i in range(1, 9)},
            "defect_norm_2": float(np.linalg.norm(defect, ord=2)),
            "defect_norm_inf": float(np.linalg.norm(defect, ord=np.inf)),
        })
        

    df_rollout = pd.DataFrame(rollout_rows)
    df_defect = pd.DataFrame(defect_rows)

    df_rollout.to_csv(out_rollout_csv, index=False)
    df_defect.to_csv(out_defect_csv, index=False)

    return df_rollout, df_defect


if __name__ == "__main__":
    base = Path(__file__).parent

    # load time-dilation parameter
    df_tdil = pd.read_csv(base / "data_p.csv")
    tdil = float(df_tdil.loc[0, "value"])

    df_rollout, df_defect = compute_iterative_rollout_defects(
        uc_csv_path=base / "data_uc.csv",
        xc_ref_path=base / "data_xd.csv",
        tdil=tdil,
        out_rollout_csv=base / "data_xc_iterative_rollout.csv",
        out_defect_csv=base / "data_defect_iterative.csv",
    )

    print("Saved:")
    print(base / "data_xc_iterative_rollout.csv")
    print(base / "data_defect_iterative.csv")