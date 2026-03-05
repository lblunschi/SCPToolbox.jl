using PyPlot

"""
Plot velocity (top) and thrust input (bottom) vs time.

Arguments
- t  :: AbstractVector  (N) time [s]
- v  :: AbstractVector  (N) velocity [m/s]
- τ  :: AbstractVector  (N) thrust/force input [N]

Keyword options
- v_label, τ_label : legend strings
- v_units, τ_units : axis units text
- figsize          : (width, height) in inches
- savepath         : if not nothing, saves figure to this path
"""
function plot_velocity_and_thrust(t, v, τ;
        v_label="Velocity",
        τ_label="Surge force",
        v_units="m/s",
        τ_units="N",
        figsize=(10, 5),
        savepath=nothing)

    fig = figure(figsize=figsize)

    # --- Top: velocity ---
    ax1 = subplot(2, 1, 1)
    ax1.plot(t, v, linewidth=2)
    ax1.set_ylabel("Velocity [$v_units]")
    ax1.grid(true, linewidth=0.3, alpha=0.5)
    ax1.legend([v_label], loc="upper left")
    ax1.set_ylim([-0.2, 0.3])
    ax1.set_axisbelow(true)

    # --- Bottom: thrust ---
    ax2 = subplot(2, 1, 2, sharex=ax1)
    ax2.plot(t, τ, linewidth=2)
    ax2.set_ylabel("Force [$τ_units]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(true, linewidth=0.3, alpha=0.5)
    ax2.legend([τ_label], loc="upper left")
    ax2.set_axisbelow(true)

    # Hide top x tick labels (like your example)
    ax1.tick_params(labelbottom=false)

    tight_layout()

    if savepath !== nothing
        save_figure(savepath, "scp")
    end

    return fig
end
""" Plot the final converged trajectory.

# Arguments
    mdl: the AUV problem parameters.
    sol: the trajectory solution.
"""
function plot_dynamics_xy(mdl::AUVProblem,
                                   t::AbstractVector{<:Real},
                                   xd::AbstractMatrix{<:Real},
                                   ud::AbstractMatrix{<:Real};
                                   algo="unknown",
                                   xc=nothing,
                                   ct_res::Int=500,
                                   u_scale::Real=0.2)::Nothing

    dt_clr = get_colormap()(1.0)
    N = size(xd, 2)

    speed = [norm(@k(xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_smap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)

    fig = create_figure((3.27, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position \$r_1\$ [m]")
    ax.set_ylabel("North position \$r_2\$ [m]")

    plt.colorbar(v_smap,
                 aspect=40,
                 label="Velocity \$\\|\\dot r\\|_2\$ [m/s]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    # Optional continuous-time dots
    if xc !== nothing
        ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
        ct_pos = T_RealMatrix(undef, 2, ct_res)
        ct_speed = T_RealVector(undef, ct_res)

        for k = 1:ct_res
            xk = sample(xc, @k(ct_τ))
            @k(ct_pos) = xk[mdl.vehicle.id_r[1:2]]
            @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
        end

        for k = 1:ct_res-1
            r, v = @k(ct_pos), @k(ct_speed)
            ax.plot(r[1], r[2],
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    alpha=0.2,
                    markerfacecolor=v_smap.to_rgba(v),
                    markeredgecolor="none",
                    clip_on=false,
                    zorder=100)
        end
    end

    # Body-frame force vectors rotated into map frame (scaled)
    acc = ud[mdl.vehicle.id_u, :]          # (4×N): [τx, τy, τz, τyaw]
    pos = xd[mdl.vehicle.id_r, :]          # (4×N): [x, y, z, ψ]

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


    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    save_figure("AUV_final_traj_xy", algo)
    return nothing
end

function plot_dynamics_xz(mdl::AUVProblem,
                                   t::AbstractVector{<:Real},
                                   xd::AbstractMatrix{<:Real},
                                   ud::AbstractMatrix{<:Real};
                                   algo="unknown",
                                   xc=nothing,
                                   ct_res::Int=500,
                                   u_scale::Real=0.2)::Nothing

    dt_clr = get_colormap()(1.0)
    N = size(xd, 2)

    speed = [norm(@k(xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_smap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)

    fig = create_figure((3.27, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position \$r_1\$ [m]")
    ax.set_ylabel("Up position \$r_2\$ [m]")

    plt.colorbar(v_smap,
                 aspect=40,
                 label="Velocity \$\\|\\dot r\\|_2\$ [m/s]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    if xc !== nothing
        ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
        ct_pos = T_RealMatrix(undef, 3, ct_res)
        ct_speed = T_RealVector(undef, ct_res)

        for k = 1:ct_res
            xk = sample(xc, @k(ct_τ))
            @k(ct_pos) = xk[mdl.vehicle.id_r[1:3]]
            @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
        end

        for k = 1:ct_res-1
            r, v = @k(ct_pos), @k(ct_speed)
            ax.plot(r[1], r[3],
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    alpha=0.2,
                    markerfacecolor=v_smap.to_rgba(v),
                    markeredgecolor="none",
                    clip_on=false,
                    zorder=100)
        end
    end

    pos = xd[mdl.vehicle.id_r, :]
    ax.plot(pos[1, :], pos[3, :],
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="\$r\$",
            zorder=100)

    acc = ud[mdl.vehicle.id_u, :]
    for k = 1:N
        base = pos[1:3, k]
        tip = base + u_scale * acc[1:3, k]
        ax.plot([base[1], tip[1]], [base[3], tip[3]],
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                label=(k==1) ? "\$a\$ (scaled)" : nothing,
                zorder=99)
    end


    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    save_figure("AUV_final_traj_xz", algo)
    return nothing
end

function plot_dynamics_yz(mdl::AUVProblem,
                                   t::AbstractVector{<:Real},
                                   xd::AbstractMatrix{<:Real},
                                   ud::AbstractMatrix{<:Real};
                                   algo="unknown",
                                   xc=nothing,
                                   ct_res::Int=500,
                                   u_scale::Real=0.2)::Nothing

    dt_clr = get_colormap()(1.0)
    N = size(xd, 2)

    speed = [norm(@k(xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_smap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)

    fig = create_figure((3.27, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("North position \$r_1\$ [m]")
    ax.set_ylabel("Up position \$r_2\$ [m]")

    plt.colorbar(v_smap,
                 aspect=40,
                 label="Velocity \$\\|\\dot r\\|_2\$ [m/s]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    if xc !== nothing
        ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
        ct_pos = T_RealMatrix(undef, 3, ct_res)
        ct_speed = T_RealVector(undef, ct_res)

        for k = 1:ct_res
            xk = sample(xc, @k(ct_τ))
            @k(ct_pos) = xk[mdl.vehicle.id_r[1:3]]
            @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
        end

        for k = 1:ct_res-1
            r, v = @k(ct_pos), @k(ct_speed)
            ax.plot(r[2], r[3],
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    alpha=0.2,
                    markerfacecolor=v_smap.to_rgba(v),
                    markeredgecolor="none",
                    clip_on=false,
                    zorder=100)
        end
    end

    pos = xd[mdl.vehicle.id_r, :]
    ax.plot(pos[2, :], pos[3, :],
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="\$r\$",
            zorder=100)

    acc = ud[mdl.vehicle.id_u, :]
    for k = 1:N
        base = pos[1:3, k]
        tip = base + u_scale * acc[1:3, k]
        ax.plot([base[2], tip[2]], [base[3], tip[3]],
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                label=(k==1) ? "\$a\$ (scaled)" : nothing,
                zorder=99)
    end


    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    save_figure("AUV_final_traj_yz", algo)
    return nothing
end

function plot_dynamics_inputs_ux_uy_uz_uyaw(mdl::AUVProblem,
                                  t::AbstractVector{<:Real},
                                  ud::AbstractMatrix{<:Real};
                                  algo="unknown",
                                  uc=nothing,
                                  tf::Real = t[end],
                                  ct_res::Int=500)::Nothing

    clr = get_colormap()(1.0)

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$u_x\$ [N]", "\$u_y\$ [N]", "\$u_z\$ [N]", "\$u_{yaw}\$ [N·m]"]
    names  = ["ux", "uy", "uz", "uyaw"]

    # Bounds
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min

    # Continuous-time time vector + samples (optional)
    ct_time = LinRange(0.0, tf, ct_res)
    ct_u = nothing
    if uc !== nothing
        ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
        ct_u = hcat([sample(uc, τ)[1:4] for τ in ct_τ]...)  # 4 x ct_res
    end

    # Discrete-time inputs (use first 4 channels)
    ud4 = ud[1:4, :]

    for i in 1:4
        ax = axs[i]
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_ylabel(labels[i])

        # Plot bounds
        plot_timeseries_bound!(ax, 0.0, tf, bnd_max, 0.0)
        plot_timeseries_bound!(ax, 0.0, tf, bnd_min, 0.0)

        # Optional continuous-time line
        if ct_u !== nothing
            ax.plot(ct_time, vec(ct_u[i, :]),
                    color=clr,
                    linewidth=2)
        end

        # Discrete-time markers + connecting line
        for visible in [true, false]
            ax.plot(visible ? t : [],
                    visible ? vec(ud4[i, :]) : [],
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

        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))

        if i < 4
            ax.set_xticklabels([])
        else
            ax.set_xlabel("Time [s]")
        end
    end

    leg = axs[1].legend(framealpha=0.8, fontsize=8, loc="upper right")
    leg.set_zorder(200)

    save_figure("AUV_inputs_4ch", algo)
    return nothing
end

function plot_position_xyzyaw_vs_time(mdl::AUVProblem,
                                   t::AbstractVector{<:Real},
                                   xd::AbstractMatrix{<:Real};
                                   algo="unknown",
                                   xc=nothing,
                                   tf::Real = t[end],
                                   ct_res::Int=500)::Nothing

    clr = get_colormap()(1.0)

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$x\$ [m]", "\$y\$ [m]", "\$z\$ [m]", "\$\\psi\$ [rad]"]
    names  = ["x", "y", "z"]

    # Discrete-time positions
    pos = xd[mdl.vehicle.id_r, :]   # (4 × N)
    pos3 = pos[1:4, :]              # (x,y,z)
    
    # Optional continuous-time sampling
    ct_time = nothing
    ct_pos = nothing
    if xc !== nothing
        ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
        ct_time = LinRange(0.0, tf, ct_res)

        ct_pos = hcat([sample(xc, τ)[mdl.vehicle.id_r[1:4]] 
                       for τ in ct_τ]...)
    end

    for i in 1:4
        ax = axs[i]

        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")

        ax.set_ylabel(labels[i])

        # Continuous-time curve (smooth line)
        if ct_pos !== nothing
            ax.plot(ct_time,
                    vec(ct_pos[i, :]),
                    color=clr,
                    linewidth=2)
        end

        # Discrete-time markers + connecting line
        ax.plot(t,
                vec(pos[i, :]),
                linestyle="-",
                color=clr,
                linewidth=1.5)

        ax.plot(t,
                vec(pos[i, :]),
                linestyle="none",
                marker="o",
                markersize=4,
                markerfacecolor=clr,
                markeredgewidth=0,
                zorder=100)

        ax.autoscale(enable=true, axis="both", tight=true)

        if i < 4
            ax.set_xticklabels([])
        else
            ax.set_xlabel("Time [s]")
        end
    end

    save_figure("AUV_position_xyz_vs_time", algo)
    return nothing
end

function plot_velocity_xyzyaw_vs_time(mdl::AUVProblem,
                                   t::AbstractVector{<:Real},
                                   xd::AbstractMatrix{<:Real};
                                   algo="unknown",
                                   xc=nothing,
                                   tf::Real = t[end],
                                   ct_res::Int=500)::Nothing

    clr = get_colormap()(1.0)

    fig = create_figure((6.5, 5.5))
    axs = [fig.add_subplot(4, 1, i) for i in 1:4]

    labels = ["\$u\$ [m/s]", "\$v\$ [m/s]", "\$w\$ [m/s]", "\$r\$ [rad/s]"]
    names  = ["x", "y", "z"]

    # Discrete-time positions
    vel = xd[mdl.vehicle.id_v, :]   # (4 × N)
    
    # Optional continuous-time sampling
    ct_time = nothing
    ct_pos = nothing
    if xc !== nothing
        ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
        ct_time = LinRange(0.0, tf, ct_res)

        ct_vel = hcat([sample(xc, τ)[mdl.vehicle.id_v] for τ in ct_τ]...)
                      
    end

    for i in 1:4
        ax = axs[i]

        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")

        ax.set_ylabel(labels[i])

        # Continuous-time curve (smooth line)
        if ct_pos !== nothing
            ax.plot(ct_time,
                    vec(ct_pos[i, :]),
                    color=clr,
                    linewidth=2)
        end

        # Discrete-time markers + connecting line
        ax.plot(t,
                vec(vel[i, :]),
                linestyle="-",
                color=clr,
                linewidth=1.5)

        ax.plot(t,
                vec(vel[i, :]),
                linestyle="none",
                marker="o",
                markersize=4,
                markerfacecolor=clr,
                markeredgewidth=0,
                zorder=100)

        ax.autoscale(enable=true, axis="both", tight=true)

        if i < 4
            ax.set_xticklabels([])
        else
            ax.set_xlabel("Time [s]")
        end
    end

    save_figure("AUV_velocity_xyz_vs_time", algo)
    return nothing
end