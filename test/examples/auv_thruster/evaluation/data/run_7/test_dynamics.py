from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from scipy.integrate import solve_ivp
from pathlib import Path
import matplotlib.pyplot as plt

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

def yaw_torque_components(
    x: np.ndarray,
    u_thr: np.ndarray,
    hydro: HydroParams,
    veh: VehicleParams,
) -> tuple[float, float, float]:
    """
    Returns:
      coriolis_like, damping_like, thruster_like
    for the yaw equation.
    """
    pos = x[0:4]
    vel = x[4:8]

    u_c, v_c, _ = get_current(pos)

    # thruster contribution
    thruster_like = (veh.thruster_allocation_matrix @ u_thr)[3]

    # split the old yaw_moment term into its two parts
    coriolis_like = (
        (vel[1] - v_c) * (vel[0] - u_c) * (hydro.added_mass_y - hydro.added_mass_x)
    )

    damping_like = (
        hydro.linear_drag_yaw + hydro.quadratic_drag_yaw * abs(vel[3])
    ) * vel[3]

    return coriolis_like, damping_like, thruster_like

def abs_smooth(z: float, eps: float = 1e-3) -> float:
    return np.sqrt(z * z + eps * eps)


def get_current(state:np.ndarray,
    yawidx=3,
    shear=0.3,
    width=0.5):
    x = state[0]
    psi = state[yawidx]

    # inertial shear in +/−y (smooth) WF
    v_i = -shear * np.tanh(x / width)

    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    # WF -> body (u_i = 0)
    u_c = spsi * v_i
    v_c = cpsi * v_i
    w_c = 0.0

    return 0, 0, 0
    #return u_c, v_c, w_c



def make_control_interpolator(t_u: np.ndarray, u_vals: np.ndarray):
    """
    Returns u(t) with linear interpolation over each control channel.
    u_vals shape: (6, M)
    """
    def u_of_t(t: float) -> np.ndarray:
        return np.array([
            np.interp(t, t_u, u_vals[i, :]) for i in range(u_vals.shape[0])
        ], dtype=float)
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
    where
      x,y,z,yaw = position states
      u,v,w,r   = body-frame surge, sway, heave, yaw rate
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
    print(csv_path)
    df = pd.read_csv(csv_path)
    t_u = df["t"].to_numpy(dtype=float)

    u_cols = [c for c in df.columns if c.startswith("u")]
    if len(u_cols) != 6:
        raise ValueError(f"Expected 6 control columns uc_1..uc_6, got {u_cols}")

    # shape (6, M)
    u_vals = np.vstack([df[c].to_numpy(dtype=float) for c in u_cols])
    return t_u, u_vals


def align_yaxis_zero(ax1, ax2):
    """
    Align zero of two y-axes (ax1 and ax2).
    """
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()

    # relative position of zero in each axis
    z1 = -y1_min / (y1_max - y1_min)
    z2 = -y2_min / (y2_max - y2_min)

    # adjust ax2 to match ax1
    if z2 != z1:
        scale = (y2_max - y2_min)
        new_min = -z1 * scale
        new_max = (1 - z1) * scale
        ax2.set_ylim(new_min, new_max)

def propagate_from_uc(
    uc_csv_path: str | Path,
    x0: np.ndarray,
    tdil: float,
    out_csv_path: str | Path = "data_xc.csv",
    n_eval: int = 50,
    torque_plot_path: str | Path = "yaw_torque_components.png",
):
    hydro = HydroParams()

    TAM = np.array([
        [-0.707, -0.707,  0.707,  0.707, 0.0, 0.0],
        [ 0.707, -0.707,  0.707, -0.707, 0.0, 0.0],
        [ 0.0,    0.0,    0.0,    0.0,   1.0, 1.0],
        [ 0.1888,-0.1888,-0.1888, 0.1888,0.0, 0.0],
    ], dtype=float)
    veh = VehicleParams(thruster_allocation_matrix=TAM)

    t_u, u_vals = load_uc_csv(uc_csv_path)
    u_of_t = make_control_interpolator(t_u, u_vals)

    t0 = float(t_u[0])
    tf = float(t_u[-1])
    t_eval = np.linspace(t0, tf, n_eval)

    sol = solve_ivp(
        fun=lambda t, x: auv_dynamics(t, x, u_of_t, hydro, veh, tdil),
        t_span=(t0, tf),
        y0=np.asarray(x0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    xc = sol.y  # shape (8, n_eval)

    tau_coriolis = np.zeros_like(sol.t)
    tau_damping = np.zeros_like(sol.t)
    tau_thruster = np.zeros_like(sol.t)

    u_current = np.zeros_like(sol.t)
    v_current = np.zeros_like(sol.t)
    u_rel = np.zeros_like(sol.t)
    v_rel = np.zeros_like(sol.t)

    for k, t in enumerate(sol.t):
        xk = xc[:, k]
        uk = u_of_t(t)

        c_k, d_k, th_k = yaw_torque_components(xk, uk, hydro, veh)
        tau_coriolis[k] = c_k
        tau_damping[k] = d_k
        tau_thruster[k] = th_k

        u_c, v_c, _ = get_current(xk[0:4])
        u_current[k] = u_c
        v_current[k] = v_c

        u_rel[k] = xk[4] - u_c
        v_rel[k] = xk[5] - v_c


    out_df = pd.DataFrame({
        "t": sol.t,
        "k": np.arange(1, len(sol.t) + 1),
        "xc_1": xc[0, :],
        "xc_2": xc[1, :],
        "xc_3": xc[2, :],
        "xc_4": xc[3, :],
        "xc_5": xc[4, :],
        "xc_6": xc[5, :],
        "xc_7": xc[6, :],
        "xc_8": xc[7, :],
        "tau_yaw_coriolis": tau_coriolis,
        "tau_yaw_damping": tau_damping,
        "tau_yaw_thruster": tau_thruster,
    })
    out_df.to_csv(out_csv_path, index=False)

        # ---- plot yaw torque components + angular velocity + relative velocities ----
    yaw_rate = xc[7, :]   # r
    u_body = xc[4, :]
    v_body = xc[5, :]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # left axis: torques
    l1, = ax1.plot(sol.t, tau_coriolis, label="Coriolis yaw torque")
    l2, = ax1.plot(sol.t, tau_damping, label="Damping yaw torque")
    l3, = ax1.plot(sol.t, tau_thruster, label="Thruster yaw torque")
    ax1.axhline(0.0, linewidth=0.8)

    ax1.set_xlabel("t")
    ax1.set_ylabel("Yaw torque [Nm]")
    ax1.set_title("Yaw torque decomposition with yaw rate and relative body velocities")
    ax1.grid(True)

    # right axis: kinematic quantities
    ax2 = ax1.twinx()
    l4, = ax2.plot(sol.t, yaw_rate, linestyle="--", label="Yaw rate r")
    l5, = ax2.plot(sol.t, u_rel, linestyle=":", label="Relative surge u_r")
    l6, = ax2.plot(sol.t, v_rel, linestyle="-.", label="Relative sway v_r")

    # optional: include absolute body velocities too, faintly
    l7, = ax2.plot(sol.t, u_body, linestyle=":", alpha=0.35, label="Body surge u")
    l8, = ax2.plot(sol.t, v_body, linestyle="-.", alpha=0.35, label="Body sway v")

    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_ylabel("Velocity [m/s] / yaw rate [rad/s]")

    # align zero points on both y-axes
    align_yaxis_zero(ax1, ax2)

    # combined legend
    lines = [l1, l2, l3, l4, l5, l6, l7, l8]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    plt.savefig(torque_plot_path, dpi=150)
    plt.show()

    return out_df


if __name__ == "__main__":
    # Example initial condition.
    # Replace with your actual initial state:
    # [x, y, z, yaw, u, v, w, yaw_rate]
    base = Path(__file__).parent

    df = pd.read_csv(base / "data_xd.csv")
    cols = [f"x{i}" for i in range(1, 9)]

    x0 = df.loc[0, cols].to_numpy(dtype=float)

    base = Path(__file__).parent
    p_data = base / "data_p.csv"
    df_tdil = pd.read_csv(p_data)

    # Replace with your optimized time-dilation parameter p[veh.id_t]
    tdil = df_tdil.value[0]
    base = Path(__file__).parent

    propagate_from_uc(
        uc_csv_path=base / "data_ud.csv",
        x0=x0,
        tdil=tdil,
        out_csv_path=base / "data_xc_new.csv",
        torque_plot_path=base / "yaw_torque_components.png",
        n_eval=50,
    )