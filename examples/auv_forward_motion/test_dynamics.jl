using Printf

include("common.jl")
include("../../models/auv.jl")
include("../../core/problem.jl")
include("../../core/scvx.jl")
include("plot_dynamics.jl")


# Same as your abs_smooth
abs_smooth(x; ϵ=1e-3) = sqrt(x^2 + ϵ^2)

"""
Real-time dynamics for the AUV: dx/dt (NOT scaled by tdil).
State x = [pos(1:4); vel(5:8)] where pos=[x,y,z,yaw], vel=[u,v,w,r]
Control uvec = [tau_x, tau_y, tau_z, tau_yaw] (4)
"""
function auv_f_real(x::AbstractVector, uvec::AbstractVector, mdl::AUVProblem)
    veh = mdl.vehicle
    hyd = mdl.hydroparams
    g = mdl.env.g

    # Current (your code currently uses 0)
    u_c = 0.0
    v_c = 0.0
    w_c = 0.0

    pos = x[veh.id_r]   # 1:4
    v   = x[veh.id_v]   # 5:8
    ut  = uvec          # 4

    f = zeros(eltype(x), 8)

    # Kinematics
    f[1] = cos(pos[4])*v[1] - sin(pos[4])*v[2]
    f[2] = sin(pos[4])*v[1] + cos(pos[4])*v[2]
    f[3] = v[3]
    f[4] = v[4]

    # Dynamics
    f[5] = (1/(hyd.mass - hyd.added_mass_x)) * (
        ut[1]
        + (v[2] - v_c) * (hyd.mass*v[4] - hyd.added_mass_y*v[4])
        + (hyd.linear_drag_x + 6 * hyd.quadratic_drag_x * abs_smooth(v[1] - u_c; ϵ=1e-3)) * (v[1] - u_c)
    )

    f[6] = (1/(hyd.mass - hyd.added_mass_y)) * (
        ut[2]
        + (v[1] - u_c) * (hyd.added_mass_x*v[4] - hyd.mass*v[4])
        + (hyd.linear_drag_y + hyd.quadratic_drag_y * abs_smooth(v[2] - v_c; ϵ=1e-3)) * (v[2] - v_c)
    )

    f[7] = (1/(hyd.mass - hyd.added_mass_z)) * (
        ut[3] + hyd.buoyancy - hyd.weight
        + (hyd.linear_drag_z + hyd.quadratic_drag_z * abs_smooth(v[3] - w_c; ϵ=1e-3)) * (v[3] - w_c)
    )

    f[8] = (1/(hyd.inertia_z - hyd.added_mass_yaw)) * (
        ut[4]
        + (v[2] - v_c) * (v[1] - u_c) * (hyd.added_mass_y - hyd.added_mass_x)
        + (hyd.linear_drag_yaw + hyd.quadratic_drag_yaw * abs_smooth(v[4]; ϵ=1e-3)) * v[4]
    )

    return f
end

struct SimpleTrajInterp
    t::Vector{Float64}
    Y::Matrix{Float64}   # (dim × N)
end

"Linear interpolation sample at τ ∈ [0,1]. Returns a vector of length dim."
function sample(itp::SimpleTrajInterp, τ::Real)
    τc = clamp(Float64(τ), 0.0, 1.0)
    tf = itp.t[end]
    tt = τc * tf

    # find interval
    k = searchsortedlast(itp.t, tt)
    if k <= 0
        return itp.Y[:, 1]
    elseif k >= length(itp.t)
        return itp.Y[:, end]
    else
        t0, t1 = itp.t[k], itp.t[k+1]
        α = (tt - t0) / (t1 - t0)
        return (1-α) .* itp.Y[:, k] .+ α .* itp.Y[:, k+1]
    end
end

"""
Simulate 30s with dt=0.1s, constant +x thrust for Tthrust seconds.
Logs each step.
"""
function thrust_x_sim_30s(mdl::AUVProblem;
    tau_x=10.0,          # [N]
    Tthrust=2.0,         # [s]
    T=30.0,              # [s]
    dt=0.1               # [s]
)
    veh = mdl.vehicle
    traj = mdl.traj

    # Initial state
    x = zeros(8)
    x[4] = 0.785398 # yaw 45 deg

    N = Int(round(T/dt)) + 1
    t = collect(range(0.0, step=dt, length=N))

    xd = zeros(8, N)
    ud = zeros(4, N)

    println("==== AUV thrust-in-x sim ====")
    @printf("T=%.1f s, dt=%.2f s, steps=%d, tau_x=%.3f N, Tthrust=%.3f s\n",
            T, dt, N, tau_x, Tthrust)

    for k in 1:N
        tk = t[k]

        # Control
        uvec = zeros(4)
        if tk <= Tthrust
            uvec[1] = tau_x
        end
        """if tk <= 2.5
            uvec[1] = 0.0
        elseif tk <= 9.5
            uvec[1] = 1.0
        elseif tk <= 14.5
            uvec[1] = -1.0
        elseif tk <= 22.0
            uvec[1] = 2.0
        else
            uvec[1] = -2.0
        end"""

        # Log into arrays
        xd[:, k] .= x
        ud[:, k] .= uvec

        # Integrate one step
        if k < N
            xdot = auv_f_real(x, uvec, mdl)
            x .+= dt .* xdot
        end
    end

    return t, xd, ud
end

mdl = AUVProblem()

t, xd, ud = thrust_x_sim_30s(mdl; tau_x=10.0, Tthrust=3.0, T=30.0, dt=0.05)

# Continuous-time (linear) reconstructions for plotting
xc = SimpleTrajInterp(t, xd)
uc = SimpleTrajInterp(t, ud)

plot_dynamics_xy(mdl, t, xd, ud; algo="scp", xc=xc)
plot_dynamics_xz(mdl, t, xd, ud; algo="scp", xc=xc)
plot_dynamics_yz(mdl, t, xd, ud; algo="scp", xc=xc)

plot_dynamics_inputs_ux_uy_uz_uyaw(mdl, t, ud; algo="scp", uc=uc)

plot_position_xyzyaw_vs_time(mdl, t, xd; algo="scp", xc=xc, tf=t[end])
plot_velocity_xyzyaw_vs_time(mdl, t, xd; algo="scp", xc=xc, tf=t[end])
plot_velocity_and_thrust(t, xd[5, :], ud[1, :];
    v_label="u (sim)",
    τ_label="τx",
    savepath="velocity_thrust")