#= AUV obstacle avoidance data structures and custom methods.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington),
                   and Autonomous Systems Laboratory (Stanford University)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

using PyPlot
using Colors

include("../utils/types.jl")
include("../core/problem.jl")
include("../core/scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" AUV vehicle parameters. """
struct AUVParameters
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_u::T_IntRange # Indices of the thrust input vector
    id_t::T_Int      # Index of time dilation
    u_max::T_Real    # [N] Maximum thrust
    u_min::T_Real    # [N] Minimum thrust
end

""" AUV hydrodynamics parameters. """
struct AUVHydrodynamicsParameters
    mass::T_Real # [kg] Mass of the vehicle
    inertia_z::T_Real # [kg*m^2] Yaw moment of inertia
    added_mass_x::T_Real # [kg] Added mass in surge
    added_mass_y::T_Real # [kg] Added mass in sway
    added_mass_z::T_Real # [kg] Added mass in heave
    added_mass_yaw::T_Real # [kg*m^2] Added mass in
    linear_drag_x::T_Real # [kg/s] Linear drag in surge
    linear_drag_y::T_Real # [kg/s] Linear drag in sway
    linear_drag_z::T_Real # [kg/s] Linear drag in heave
    linear_drag_yaw::T_Real # [kg*m^2/s] Linear drag in yaw
    quadratic_drag_x::T_Real # [kg/m] Quadratic drag in surge
    quadratic_drag_y::T_Real # [kg/m] Quadratic drag in sway
    quadratic_drag_z::T_Real # [kg/m] Quadratic drag in heave
    quadratic_drag_yaw::T_Real # [kg*m] Quadratic drag in yaw
    buoyancy::T_Real # [N] buoyancy force
    weight::T_Real # [N] Weight force
end
""" AUV hydrodynamics parameters. """
struct AUVHydrodynamicsParameters
    mass::T_Real
    inertia_z::T_Real
    added_mass_x::T_Real
    added_mass_y::T_Real
    added_mass_z::T_Real
    added_mass_yaw::T_Real
    linear_drag_x::T_Real
    linear_drag_y::T_Real
    linear_drag_z::T_Real
    linear_drag_yaw::T_Real
    quadratic_drag_x::T_Real
    quadratic_drag_y::T_Real
    quadratic_drag_z::T_Real
    quadratic_drag_yaw::T_Real
    buoyancy::T_Real
    weight::T_Real
end

# Keyword constructor (outer constructor)
function AUVHydrodynamicsParameters(; mass, inertia_z,
    added_mass_x, added_mass_y, added_mass_z, added_mass_yaw,
    linear_drag_x, linear_drag_y, linear_drag_z, linear_drag_yaw,
    quadratic_drag_x, quadratic_drag_y, quadratic_drag_z, quadratic_drag_yaw,
    buoyancy, weight)

    return AUVHydrodynamicsParameters(
        mass, inertia_z,
        added_mass_x, added_mass_y, added_mass_z, added_mass_yaw,
        linear_drag_x, linear_drag_y, linear_drag_z, linear_drag_yaw,
        quadratic_drag_x, quadratic_drag_y, quadratic_drag_z, quadratic_drag_yaw,
        buoyancy, weight
    )
end
""" AUV flight environment. """
struct AUVEnvironmentParameters
    g::T_RealVector          # [m/s^2] Gravity vector
    obs::Vector{T_Ellipsoid} # Obstacles (ellipsoids)
    n_obs::T_Int             # Number of obstacles
end

""" Trajectory parameters. """
struct AUVTrajectoryParameters
    r0::T_RealVector # Initial position
    rf::T_RealVector # Terminal position
    v0::T_RealVector # Initial velocity
    vf::T_RealVector # Terminal velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
    γ::T_Real        # Minimum-time vs. minimum-energy tradeoff
end

""" AUV trajectory optimization problem parameters all in one. """
struct AUVProblem
    vehicle::AUVParameters        # The ego-vehicle
    env::AUVEnvironmentParameters # The environment
    traj::AUVTrajectoryParameters # The trajectory
    hydroparams::AUVHydrodynamicsParameters # The hydrodynamics parameters
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Constructor for the environment.

# Arguments
    gnrm: gravity vector norm.
    obs: array of obstacles (ellipsoids).

# Returns
    env: the environment struct.
"""
function AUVEnvironmentParameters(
    gnrm::T_Real,
    obs::Vector{T_Ellipsoid})::AUVEnvironmentParameters

    # Derived values
    g = zeros(3)
    g[end] = -gnrm
    n_obs = length(obs)

    env = AUVEnvironmentParameters(g, obs, n_obs)

    return env
end

""" Constructor for the AUV problem.

# Returns
    mdl: the AUV problem.
"""
function AUVProblem()::AUVProblem

    # >> AUV <<
    id_r = 1:4 # Position indices of the state vector (x, y, z, yaw)
    id_v = 5:8 # Velocity indices of the state vector (vx, vy, vz, yaw_rate)
    id_u = 1:4 # Indices of the thrust input vector (taux, tauy, tauz, tauyaw)
    id_t = 1    # Index of time dilation
    u_max = 39.2266 # [N] Maximum thrust) TODO define rigourously, for now use 4 kg f
    u_min = -39.2266  # [N] Minimum thrust
    auv = AUVParameters(id_r, id_v, id_u, id_t,
                               u_max, u_min)

    # >> Environment <<
    g = 9.81
    obs = T_Ellipsoid[] # No obstacles
    env = AUVEnvironmentParameters(g, obs)

    # >> Trajectory <<
    r0 = zeros(4) # Initial position (x, y, z, yaw)
    r0[3] = -0.5  # Initial depth
    rf = zeros(4)  # Terminal position (x, y, z, yaw)
    rf[1:3] = [2.5; 6.0; -5.0] # Terminal position (x, y, z)
    v0 = zeros(4) # Initial velocity (u, v, w, yaw_rate)
    v0[1] = 0.5 # Initial surge velocity
    vf = zeros(4) # Terminal velocity (u, v, w, yaw_rate)
    tf_min = 0.0 # Minimum flight time
    tf_max = 30.0 # Maximum flight time
    γ = 0.0 # Minimum-time vs. minimum-energy tradeoff (0: min time, 1: min energy)
    traj = AUVTrajectoryParameters(r0, rf, v0, vf, tf_min, tf_max, γ)

    # >> Hydrodynamics << from https://flex.flinders.edu.au/file/27aa0064-9de2-441c-8a17-655405d5fc2e/1/ThesisWu2018.pdf
    hydro = AUVHydrodynamicsParameters(
        mass=11.5, # [kg] Mass of the vehicle
        inertia_z=0.16, # [kg*m^2] Yaw moment of inertia
        added_mass_x=-5.5, # [kg] Added mass in surge
        added_mass_y=-12.7, # [kg] Added mass in sway
        added_mass_z=-14.57, # [kg] Added mass in heave
        added_mass_yaw=-0.12, # [kg*m^2/rad] Added mass in yaw
        linear_drag_x=-4.03, # [Ns/m] Linear drag in surge
        linear_drag_y=-6.22, # [Ns/m] Linear drag in sway
        linear_drag_z=-5.18, # [Ns/m] Linear drag in heave
        linear_drag_yaw=-0.07, # [Ns/rad] Linear drag in yaw
        quadratic_drag_x=-18.18, # [Ns²/m²] Quadratic drag in surge
        quadratic_drag_y=-21.66, # [Ns²/m²] Quadratic drag in sway
        quadratic_drag_z=-36.99, # [Ns²/m²] Quadratic drag in heave
        quadratic_drag_yaw=-1.55, # [Ns²/rad²] Quadratic drag in yaw
        buoyancy=114.8, # [N] buoyancy force (buoyancy = mass*gravity)
        weight=112.8)  # [N] Weight force (weight = mass*gravity)


    mdl = AUVProblem(auv, env, traj, hydro)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Plot the trajectory evolution through SCP iterations.

# Arguments
    mdl: the AUV problem parameters.
    history: SCP iteration data history.
"""
function plot_trajectory_history(mdl::AUVProblem,
                                 history::SCPHistory)::Nothing

    # Common values
    num_iter = length(history.subproblems)
    algo = history.subproblems[1].algo
    cmap = get_colormap()
    cmap_offset = 0.1
    alph_offset = 0.3

    fig = create_figure((2.58, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position \$r_1\$ [m]")
    ax.set_ylabel("North position \$r_2\$ [m]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    # ..:: Draw the trajectories ::..
    for i = 0:num_iter
        # Extract values for the trajectory at iteration i
        if i==0
            trj = history.subproblems[1].ref
            alph = alph_offset
            clr = parse(RGB, "#356397")
            clr = rgb2pyplot(clr, a=alph)
            shp = "X"
        else
            trj = history.subproblems[i].sol
            f = (off) -> (i-1)/(num_iter-1)*(1-off)+off
            alph = f(alph_offset)
            clr = (cmap(f(cmap_offset))..., alph)
            shp = "o"
        end
        pos = trj.xd[mdl.vehicle.id_r, :]
        x, y = pos[1, :], pos[2, :]

        label = nothing
        if i == 0
            label = "Initial \$r\$"
        elseif i == num_iter
            label = "Converged \$r\$"
        end

        ax.plot(x, y,
                linestyle="none",
                marker=shp,
                markersize=5,
                markerfacecolor=clr,
                markeredgecolor=(1, 1, 1, alph),
                markeredgewidth=0.3,
                label=label,
                zorder=100)
    end

    ax.set_xticks(-0.5:1:5)

    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (-0.5, missing, -0.5, 6.5))

    save_figure("AUV_traj_iters", algo)

    return nothing
end

""" Plot the final converged trajectory.

# Arguments
    mdl: the AUV problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory_x_y(mdl::AUVProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = get_colormap()(1.0)
    N = size(sol.xd, 2)

    max_err = 0.0
    
    speed = [norm(@k(sol.xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_cmap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)
    u_scale = 0.2

    fig = create_figure((3.27, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position \$r_1\$ [m]")
    ax.set_ylabel("North position \$r_2\$ [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity \$\\|\\dot r\\|_2\$ [m/s]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    # Use solver's actual normalized knot times
    τk = sol.td                 # length N
    posd = sol.xd[mdl.vehicle.id_r[1:2], :]

    # Plot xc evaluated exactly at the discrete knots
    for k in 1:length(τk)
        xk = sample(sol.xc, τk[k])
        rk = xk[mdl.vehicle.id_r[1:2]]
        ax.plot(rk[1], rk[2],
                linestyle="none",
                marker="x",
                markersize=4,
                alpha=0.8,
                markerfacecolor="none",
                markeredgecolor="tab:blue",
                zorder=150)
    end

    errs = zeros(length(τk))
    for k in 1:length(τk)
        errs[k] = maximum(abs.(sample(sol.xc, τk[k]) .- sol.xd[:,k]))
    end
    @show maximum(errs) argmax(errs)

    @show sol.xd[:,end]
    @show sample(sol.xc, 1.0)
    @show sol.xd[:,1]
    @show sample(sol.xc, 0.0)
    # ..:: Draw the discrete-time positions trajectory ::..
    pos = sol.xd[mdl.vehicle.id_r, :]
    x, y = pos[1, :], pos[2, :]
    ax.plot(x, y,
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="\$r\$",
            zorder=100)

    # ..:: Draw the acceleration vector ::..
    # Body-frame force vectors rotated into map frame (scaled)
    
    acc = sol.ud[mdl.vehicle.id_u, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    for k = 1:N
        base = pos[1:2, k]
        ψ = pos[4, k]

        # body-frame horizontal force
        fb = acc[1:2, k]                   # [τx; τy]

        # rotate body -> inertial/map
        c, s = cos(ψ), sin(ψ)
        fi_x = c * fb[1] - s * fb[2]
        fi_y = s * fb[1] + c * fb[2]

        nrm = sqrt(fi_x^2 + fi_y^2)
        dir = nrm > 1e-9 ? [fi_x, fi_y] ./ nrm : [0.0, 0.0]
        tip = base .+ u_scale .* dir

        ax.plot([base[1], tip[1]], [base[2], tip[2]],
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                label=(k==1) ? "\$\\tau\$ (scaled, body→map)" : nothing,
                zorder=99)
    end

    ax.set_xticks(-0.5:1:5)

    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (-0.5, missing, -0.5, 6.5))

    save_figure("AUV_final_traj_xy", algo)

    return nothing
end

""" Plot the final converged trajectory.

# Arguments
    mdl: the AUV problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory_x_z(mdl::AUVProblem,
                               sol::SCPSolution)::Nothing
    @show typeof(sol.xc)
    try
        @show sol.xc.t[1] sol.xc.t[end]
    catch
    end
    # Common values
    algo = sol.algo
    dt_clr = get_colormap()(1.0)
    N = size(sol.xd, 2)
    speed = [norm(@k(sol.xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_cmap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)
    u_scale = 0.2

    fig = create_figure((3.27, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position \$r_1\$ [m]")
    ax.set_ylabel("Up position \$r_2\$ [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity \$\\|\\dot r\\|_2\$ [m/s]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    # ..:: Draw the final continuous-time position trajectory ::..
    # Collect the continuous-time trajectory data
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_pos = T_RealMatrix(undef, 3, ct_res)
    ct_speed = T_RealVector(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, @k(ct_τ))
        @k(ct_pos) = xk[mdl.vehicle.id_r[1:3]]
        @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res-1
        r, v = @k(ct_pos), @k(ct_speed)
        x, z = r[1], r[3]
        ax.plot(x, z,
                linestyle="none",
                marker="o",
                markersize=4,
                alpha=0.2,
                markerfacecolor=v_cmap.to_rgba(v),
                markeredgecolor="none",
                clip_on=false,
                zorder=100)
    end


    # ..:: Draw the discrete-time positions trajectory ::..
    pos = sol.xd[mdl.vehicle.id_r, :]
    x, z = pos[1, :], pos[3, :]
    ax.plot(x, z,
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="\$r\$",
            zorder=100)

    # ..:: Draw the acceleration vector ::..
    acc = sol.ud[mdl.vehicle.id_u, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    for k = 1:N
        base = pos[1:3, k]
        tip = base+u_scale*acc[1:3, k]
        x = [base[1], tip[1]]
        z = [base[3], tip[3]]
        ax.plot(x, z,
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                label=(k==1) ? "\$a\$ (scaled)" : nothing,
                zorder=99)
    end

    ax.set_xticks(-0.5:1:5)

    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (-0.5, missing, -10.0, 0.0))

    save_figure("AUV_final_traj_xz", algo)

    return nothing
end

""" Plot the final converged trajectory.

# Arguments
    mdl: the AUV problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory_y_z(mdl::AUVProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = get_colormap()(1.0)
    N = size(sol.xd, 2)
    speed = [norm(@k(sol.xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_cmap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)
    u_scale = 0.2

    fig = create_figure((3.27, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("North position \$r_1\$ [m]")
    ax.set_ylabel("Up position \$r_2\$ [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity \$\\|\\dot r\\|_2\$ [m/s]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    # ..:: Draw the final continuous-time position trajectory ::..
    # Collect the continuous-time trajectory data
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_pos = T_RealMatrix(undef, 3, ct_res)
    ct_speed = T_RealVector(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, @k(ct_τ))
        @k(ct_pos) = xk[mdl.vehicle.id_r[1:3]]
        @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res-1
        r, v = @k(ct_pos), @k(ct_speed)
        y, z = r[2], r[3]
        ax.plot(y, z,
                linestyle="none",
                marker="o",
                markersize=4,
                alpha=0.2,
                markerfacecolor=v_cmap.to_rgba(v),
                markeredgecolor="none",
                clip_on=false,
                zorder=100)
    end


    # ..:: Draw the discrete-time positions trajectory ::..
    pos = sol.xd[mdl.vehicle.id_r, :]
    y, z = pos[2, :], pos[3, :]
    ax.plot(y, z,
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="\$r\$",
            zorder=100)

    # ..:: Draw the acceleration vector ::..
    acc = sol.ud[mdl.vehicle.id_u, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    for k = 1:N
        base = pos[1:3, k]
        tip = base+u_scale*acc[1:3, k]
        y = [base[2], tip[2]]
        z = [base[3], tip[3]]
        ax.plot(y, z,
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                label=(k==1) ? "\$a\$ (scaled)" : nothing,
                zorder=99)
    end

    ax.set_xticks(-0.5:1:5)

    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (-0.5, missing, -10.0, 0.0))

    save_figure("AUV_final_traj_yz", algo)

    return nothing
end

""" Plot the acceleration input norm.

# Arguments
    mdl: the AUV problem parameters.
    sol: the trajectory solution.
"""
function plot_inputs_ux_uy_uz_uyaw(mdl::AUVProblem,
                                  sol::SCPSolution)::Nothing

    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = sol.p[mdl.vehicle.id_t]

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$u_x\$ [N]", "\$u_y\$ [N]", "\$u_z\$ [N]", "\$u_{yaw}\$ [N·m]"]
    names  = ["ux", "uy", "uz", "uyaw"]

    # Time vectors
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ * tf

    time = sol.td * tf

    # Bounds
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min

    # Continuous-time inputs (4 x ct_res)
    ct_u = hcat([sample(sol.uc, τ)[1:4] for τ in ct_τ]...)

    # Discrete-time inputs (4 x N)
    ud = sol.ud[1:4, :]

    for i in 1:4
        ax = axs[i]
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_ylabel(labels[i])

        # Plot bounds
        plot_timeseries_bound!(ax, 0.0, tf, bnd_max, 0.0)  # upper
        plot_timeseries_bound!(ax, 0.0, tf, bnd_min, 0.0)  # lower

        # Continuous-time line
        ax.plot(ct_time, vec(ct_u[i, :]),
                color=clr,
                linewidth=2)

        # Discrete-time markers + connecting line
        for visible in [true, false]
            ax.plot(visible ? time : [],
                    visible ? vec(ud[i, :]) : [],
                    linestyle=visible ? "none" : "-",
                    color=visible ? nothing : clr,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    markeredgewidth=0,
                    markerfacecolor=clr,
                    zorder=100 - T_Int(!visible) * 200,
                    clip_on=!visible,
                    label=visible ? nothing : names[i])
        end

        # X formatting
        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))
        ax.set_ylim((-5.0, 5.0))
        ax.set_xticks(LinRange(0, tf_max, 6))


        if i < 4
            ax.set_xticklabels([])  # hide tick labels on upper plots
        else
            ax.set_xlabel("Time [s]")
        end
    end

    # Put one legend on the top axis (cleaner)
    leg = axs[1].legend(framealpha=0.8, fontsize=8, loc="upper right")
    leg.set_zorder(200)

    save_figure("AUV_inputs_4ch", algo)

    return nothing
end

function plot_position_xyzyaw_vs_time(mdl::AUVProblem,
                                  sol::SCPSolution)::Nothing


    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = sol.p[mdl.vehicle.id_t]

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$u_x\$ [N]", "\$u_y\$ [N]", "\$u_z\$ [N]", "\$u_{yaw}\$ [N·m]"]
    names  = ["ux", "uy", "uz", "uyaw"]

    # Time vectors
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ * tf

    time = sol.td * tf

    # Bounds
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min

    # Continuous-time inputs (4 x ct_res)
    ct_u = hcat([sample(sol.xc, τ)[1:4] for τ in ct_τ]...)

    # Discrete-time inputs (4 x N)
    ud = sol.xd[1:4, :]

    for i in 1:4
        ax = axs[i]
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_ylabel(labels[i])

        # Plot bounds
        plot_timeseries_bound!(ax, 0.0, tf, bnd_max, 0.0)  # upper
        plot_timeseries_bound!(ax, 0.0, tf, bnd_min, 0.0)  # lower

        # Continuous-time line
        ax.plot(ct_time, vec(ct_u[i, :]),
                color=clr,
                linewidth=2)

        # Discrete-time markers + connecting line
        for visible in [true, false]
            ax.plot(visible ? time : [],
                    visible ? vec(ud[i, :]) : [],
                    linestyle=visible ? "none" : "-",
                    color=visible ? nothing : clr,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    markeredgewidth=0,
                    markerfacecolor=clr,
                    zorder=100 - T_Int(!visible) * 200,
                    clip_on=!visible,
                    label=visible ? nothing : names[i])
        end

        # X formatting
        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))
        ax.set_ylim((-5.0, 5.0))
        ax.set_xticks(LinRange(0, tf_max, 6))


        if i < 4
            ax.set_xticklabels([])  # hide tick labels on upper plots
        else
            ax.set_xlabel("Time [s]")
        end
    end

    # Put one legend on the top axis (cleaner)
    leg = axs[1].legend(framealpha=0.8, fontsize=8, loc="upper right")
    leg.set_zorder(200)

    save_figure("AUV_position_4ch", algo)

    return nothing
end

function plot_velocity_xyzyaw_vs_time(mdl::AUVProblem,
                                  sol::SCPSolution)::Nothing


    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = sol.p[mdl.vehicle.id_t]

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$u_x\$ [N]", "\$u_y\$ [N]", "\$u_z\$ [N]", "\$u_{yaw}\$ [N·m]"]
    names  = ["ux", "uy", "uz", "uyaw"]

    # Time vectors
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ * tf

    time = sol.td * tf

    # Bounds
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min

    # Continuous-time inputs (4 x ct_res)
    ct_u = hcat([sample(sol.xc, τ)[5:8] for τ in ct_τ]...)

    # Discrete-time inputs (4 x N)
    ud = sol.xd[5:8, :]

    for i in 1:4
        ax = axs[i]
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_ylabel(labels[i])

        # Plot bounds
        plot_timeseries_bound!(ax, 0.0, tf, bnd_max, 0.0)  # upper
        plot_timeseries_bound!(ax, 0.0, tf, bnd_min, 0.0)  # lower

        # Continuous-time line
        ax.plot(ct_time, vec(ct_u[i, :]),
                color=clr,
                linewidth=2)

        # Discrete-time markers + connecting line
        for visible in [true, false]
            ax.plot(visible ? time : [],
                    visible ? vec(ud[i, :]) : [],
                    linestyle=visible ? "none" : "-",
                    color=visible ? nothing : clr,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    markeredgewidth=0,
                    markerfacecolor=clr,
                    zorder=100 - T_Int(!visible) * 200,
                    clip_on=!visible,
                    label=visible ? nothing : names[i])
        end

        # X formatting
        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))
        ax.set_ylim((-5.0, 5.0))
        ax.set_xticks(LinRange(0, tf_max, 6))


        if i < 4
            ax.set_xticklabels([])  # hide tick labels on upper plots
        else
            ax.set_xlabel("Time [s]")
        end
    end

    # Put one legend on the top axis (cleaner)
    leg = axs[1].legend(framealpha=0.8, fontsize=8, loc="upper right")
    leg.set_zorder(200)

    save_figure("AUV_velocity_4ch", algo)

    return nothing
end