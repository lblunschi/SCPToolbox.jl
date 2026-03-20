"""
Starship landing plots.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

using PyPlot
using Colors

""" Get a plotting colormap.

The colormap is normalized to the [0, 1] interval.

# Returns
    cmap: a colormap object that can be queried for RGB color.
"""
function get_colormap()::PyPlot.PyObject
    cmap = plt.get_cmap("inferno_r")
    nrm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    clr_query = matplotlib.cm.ScalarMappable(norm=nrm, cmap=cmap)
    cmap = (f::Real) -> clr_query.to_rgba(f)[1:3]
    return cmap
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Plot the trajectory evolution through SCP iterations.

# Arguments
    mdl: the AUVSimplified problem parameters.
    history: SCP iteration data history.
"""
function plot_trajectory_history(mdl::AUVSimplifiedProblem,
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

    #set_axis_equal(ax, (-0.5, missing, -0.5, 6.5))

    save_figure("AUVSimplified_traj_iters", algo)

    return nothing
end

""" Plot the final converged trajectory.

# Arguments
    mdl: the AUVSimplified problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory_x_y(mdl::AUVSimplifiedProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = get_colormap()(1.0)
    N = size(sol.xd, 2)

    max_err = 0.0
    
    speed = [norm(sol.xd[mdl.vehicle.id_v, k]) for k=1:N]
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
    thrust = sol.ud[mdl.vehicle.id_u, :]
    acc = thrust
   
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

    #set_axis_equal(ax, (-0.5, missing, -0.5, 6.5))

    save_figure("AUVSimplified_final_traj_xy", algo)

    return nothing
end

""" Plot the final converged trajectory.

# Arguments
    mdl: the AUVSimplified problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory_x_z(mdl::AUVSimplifiedProblem,
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
    speed = [norm((sol.xd[mdl.vehicle.id_v, k])) for k=1:N]
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
    ct_τ = collect(LinRange(0.0, 1.0, ct_res))
    ct_pos = Matrix{Float64}(undef, 3, ct_res)
    ct_speed = Vector{Float64}(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, (ct_τ[k]))
        ct_pos[:,k] = xk[mdl.vehicle.id_r[1:3]]
        ct_speed[k] = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res
        r, v = ct_pos[:,k], ct_speed[k]
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
    thrust = sol.ud[mdl.vehicle.id_u, :]
    acc = thrust
  
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

    #set_axis_equal(ax, (-0.5, missing, -10.0, 0.0))

    save_figure("AUVSimplified_final_traj_xz", algo)

    return nothing
end

""" Plot the final converged trajectory.

# Arguments
    mdl: the AUVSimplified problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory_y_z(mdl::AUVSimplifiedProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = get_colormap()(1.0)
    N = size(sol.xd, 2)
    speed = [norm(sol.xd[mdl.vehicle.id_v, k]) for k=1:N]
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
    ct_τ = collect(LinRange(0.0, 1.0, ct_res))
    ct_pos = Matrix{Float64}(undef, 3, ct_res)
    ct_speed = Vector{Float64}(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, ct_τ[k])
        ct_pos[:,k] = xk[mdl.vehicle.id_r[1:3]]
        ct_speed[k] = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res
        r, v = ct_pos[:,k], ct_speed[k]
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
    thrust = sol.ud[mdl.vehicle.id_u, :]
    acc = thrust
   
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

    #set_axis_equal(ax, (-0.5, missing, -10.0, 0.0))

    save_figure("AUVSimplified_final_traj_yz", algo)

    return nothing
end

""" Plot the acceleration input norm.

# Arguments
    mdl: the AUVSimplified problem parameters.
    sol: the trajectory solution.
"""
function plot_inputs_ux_uy_uz_uyaw(mdl::AUVSimplifiedProblem,
                                  sol::SCPSolution)::Nothing

    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = mdl.traj.tf

    fig = create_figure((6.5, 5.5))
    
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$t_1\$ [N]", "\$t_2\$ [N]", "\$t_3\$ [N]", "\$t_4\$ [N]"]
    names  = ["t1", "t2", "t3", "t4"]

    # Time vectors
    ct_res = 500
    ct_τ = collect(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ * tf

    time = sol.td * tf

    # Bounds
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min

    # Continuous-time inputs (6 x ct_res)
    ct_u = hcat([sample(sol.uc, τ)[1:4] for τ in ct_τ]...)

    # Discrete-time inputs (6 x N)
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
                    zorder=100 - Int(!visible) * 200,
                    clip_on=!visible,
                    label=visible ? nothing : names[i])
        end

        # X formatting
        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))
        max_y = maximum(ud[i,:])
        min_y = minimum(ud[i,:])
        ax.set_ylim((min_y-0.05, max_y+0.05))
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
    fig.tight_layout()

    save_figure("AUVSimplified_inputs_6ch", algo)

    return nothing
end

function plot_position_xyzyaw_vs_time(mdl::AUVSimplifiedProblem,
                                  sol::SCPSolution)::Nothing


    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = mdl.traj.tf

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$x\$ [m]", "\$y\$ [m]", "\$z\$ [m]", "\${yaw}\$ [rad]"]
    names  = ["x", "y", "z", "yaw"]

    # Time vectors
    ct_res = 500
    ct_τ = collect(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ * tf

    time = sol.td * tf



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
                    zorder=100 - Int(!visible) * 200,
                    clip_on=!visible,
                    label=visible ? nothing : names[i])
        end

        # X formatting
        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))
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

    save_figure("AUVSimplified_position_4ch", algo)

    return nothing
end

function plot_velocity_xyzyaw_vs_time(mdl::AUVSimplifiedProblem,
                                  sol::SCPSolution)::Nothing


    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = mdl.traj.tf

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$u_x\$ [N]", "\$u_y\$ [N]", "\$u_z\$ [N]", "\$u_{yaw}\$ [N·m]"]
    names  = ["ux", "uy", "uz", "uyaw"]

    # Time vectors
    ct_res = 500
    ct_τ = collect(LinRange(0.0, 1.0, ct_res))
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
                    zorder=100 - Int(!visible) * 200,
                    clip_on=!visible,
                    label=visible ? nothing : names[i])
        end

        # X formatting
        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))
        ax.set_ylim((-0.5, 0.5))
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

    save_figure("AUVSimplified_velocity_4ch", algo)

    return nothing
end

function plot_dynamics_inputs_ux_uy_uz_uyaw_energy(mdl::AUVSimplifiedProblem,
                                  t::AbstractVector{<:Real},
                                  xd::AbstractMatrix{<:Real},
                                  ud::AbstractMatrix{<:Real};
                                  algo="unknown",
                                  uc=nothing,
                                  tf::Real = t[end],
                                  ct_res::Int=500)::Nothing

    clr = get_colormap()(1.0)

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(5, 1, i) for i in 1:5]

    labels = ["\$u_x\$ [N]", "\$u_y\$ [N]", "\$u_z\$ [N]", "\$u_{yaw}\$ [N·m]", "Energy"]
    names  = ["ux", "uy", "uz", "uyaw", "e"]

    # Bounds
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min

    # Continuous-time time vector + samples (optional)
    ct_time = LinRange(0.0, tf, ct_res)
    ct_u = nothing
    if uc !== nothing
        ct_τ = collect(LinRange(0.0, 1.0, ct_res))
        ct_u = hcat([sample(uc, τ)[1:4] for τ in ct_τ]...)  # 4 x ct_res
    end

    # Discrete-time inputs (use first 4 channels)
    ud4 = ud[1:4, :]

    for i in 1:5
        ax = axs[i]
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_ylabel(labels[i])
        if i < 5
            # Plot bounds
            plot_timeseries_bound!(ax, 0.0, tf, bnd_max, 0.0)
            plot_timeseries_bound!(ax, 0.0, tf, bnd_min, 0.0)
                
            # Optional continuous-time line
            if ct_u !== nothing
                ax.plot(ct_time, vec(ct_u[i, :]),
                        color=clr,
                        linewidth=2)
            end

            for visible in [true, false]
                ax.plot(t,
                        vec(ud4[i, :]),
                        linestyle=visible ? "none" : "-",
                        color=visible ? nothing : clr,
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        markeredgewidth=0,
                        markerfacecolor=clr,
                        zorder=100 - Int(!visible) * 200,
                        clip_on=!visible,
                        label=visible ? nothing : names[i])
            end

            max_y = maximum(ud4[i,:])
            min_y = minimum(ud4[i,:])
            ax.set_ylim((min_y-0.05, max_y+0.05))
            tf_max = round(tf, digits=5)
            ax.set_xlim((0.0, tf_max))
        else
            # Plot energy

            # Discrete-time markers + connecting line
            for visible in [true, false]
                ax.plot(t,
                        vec(xd[9, :]),
                        linestyle=visible ? "none" : "-",
                        color=visible ? nothing : clr,
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        markeredgewidth=0,
                        markerfacecolor=clr,
                        zorder=100 - Int(!visible) * 200,
                        clip_on=!visible,
                        label=visible ? nothing : names[i])
            end

            max_y = maximum(xd[9,:])
            min_y = minimum(xd[9,:])
            ax.set_ylim((min_y-0.05, max_y+0.05))
            tf_max = round(tf, digits=5)
            ax.set_xlim((0.0, tf_max))
        end


        if i < 5
            ax.set_xticklabels([])
        else
            ax.set_xlabel("Time [s]")
        end
    end

    leg = axs[1].legend(framealpha=0.8, fontsize=8, loc="upper right")
    leg.set_zorder(200)

    save_figure("AUVSimplified_inputs_4ch", algo)
    return nothing
end
