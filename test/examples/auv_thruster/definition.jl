#= AUV landing flip maneuver problem definition.

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
this program.  If not, see <https://www.gnu.org/licenses/>. =#

using JuMP
using ECOS
using Printf
using ForwardDiff
using LinearAlgebra
using Symbolics
# ..:: Methods ::..

function define_problem!(pbm::TrajectoryProblem, algo::Symbol)::Nothing

    set_dims!(pbm)
    set_scale!(pbm)
    set_cost!(pbm)
    set_dynamics!(pbm)
    set_convex_constraints!(pbm)
    set_nonconvex_constraints!(pbm, algo)
    set_bcs!(pbm)

    set_guess!(pbm)
    

    return nothing
end

function set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 8, 6, 1) # nx, nu, np

    return nothing
end

function set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl
    veh = mdl.vehicle
    traj = mdl.traj

    advise! = problem_advise_scale!

    # States
    x0 = traj.r0
    xf = traj.rf
    min_x = minimum([x0[1],xf[1]]) - 2
    max_x = maximum([x0[1],xf[1]]) + 2
    min_y = minimum([x0[2],xf[2]]) - 2
    max_y = maximum([x0[2],xf[2]]) + 2
    min_z = minimum([x0[3],xf[3]]) - 2
    max_z = min(0, maximum([x0[3],xf[3]]) + 2)
    # Position
    advise!(pbm, :state, veh.id_r[1], (min_x, max_x))
    advise!(pbm, :state, veh.id_r[2], (min_y, max_y))
    advise!(pbm, :state, veh.id_r[3], (min_z, max_z))
    advise!(pbm, :state, veh.id_r[4], (-3.1415, 3.1415)) # TODO check how to scale yaw
    # Velocity

    # advise!(pbm, :state, veh.id_v[1], (veh.v_min_u, veh.v_max))
    # advise!(pbm, :state, veh.id_v[2:3], (veh.v_min, veh.v_max))
    advise!(pbm, :state, veh.id_v[1], (-0.05, 0.2))
    advise!(pbm, :state, veh.id_v[2], (-0.05,0.05))
    advise!(pbm, :state, veh.id_v[3], (-0.2, 0.2))
    advise!(pbm, :state, veh.id_v[4], (-0.1, 0.1)) # TODO check how to scale yaw rate
    # Inputs
    advise!(pbm, :input, veh.id_u, (veh.u_min, veh.u_max))
    # Parameters
    advise!(pbm, :parameter, veh.id_t, (traj.tf_min, traj.tf_max))

    return nothing
end


""" Compute the initial trajectory guess.

This uses a simple bang-bang control strategy for the flip maneuver. Once
AUV is upright, convex optimization is used to find the terminal descent
trajectory by approximatin AUV as a double-integrator (no attitude, no
aerodynamics).

Args:
* `N`: the number of discrete-time grid nodes.
* `pbm`: the trajectory problem structure.

Returns:
* `x_guess`: the state trajectory initial guess.
* `u_guess`: the input trajectory initial guess.
* `p_guess`: the parameter vector initial guess.
"""
function auv_initial_guess(
    N::Int,
    pbm::TrajectoryProblem,
)::Tuple{RealMatrix,RealMatrix,RealVector}
    if pbm.mdl.traj.use_guess == 0
        @printf("Computing initial guess .")

        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        g = pbm.mdl.env.g
        hyd = pbm.mdl.hydroparams

        p_guess = zeros(1)
        p_guess[1] = (traj.tf_min + traj.tf_max)/2
        # TEST TODO remove
        p_guess[1] = traj.tf_guess
        # State guess
        x0 = zeros(pbm.nx)
        xf = zeros(pbm.nx)
        x0[veh.id_r] = traj.r0
        xf[veh.id_r] = traj.rf
        x0[veh.id_v] = traj.v0
        xf[veh.id_v] = traj.vf

        dx = xf[1] - x0[1]
        dy = xf[2] - x0[2]
        yaw = atan(dy, dx) 
        xf[4] = yaw

        x_guess = straightline_interpolate(x0, xf, N)
        u_guess = zeros(pbm.nu, N)
        dist = norm(xf[1:2]-x0[1:2])
        forward_vel = dist/p_guess[1]
        dist_up = xf[3]-x0[3]
        up_vel = dist_up/p_guess[1]
        for k in 1:N
            x_guess[5, k] = forward_vel
            x_guess[7, k] = up_vel

            # get current at the desired position
            u_c, v_c, w_c = get_current(x_guess[:,k])
            # compute relative velocity
            rel_forward_vel = forward_vel-u_c
            rel_sideway_vel = 0.0 - v_c

            thrust_forwad = -(hyd.linear_drag_x + hyd.quadratic_drag_x*abs_smooth(rel_forward_vel))*rel_forward_vel

            thrust_sideway = -(hyd.linear_drag_y + hyd.quadratic_drag_y*abs_smooth(rel_sideway_vel))*rel_sideway_vel

            thrust_up = (- hyd.buoyancy + hyd.weight - (hyd.linear_drag_z + hyd.quadratic_drag_z*abs_smooth(up_vel))*up_vel)

            thrust_yaw = - (hyd.added_mass_y - hyd.added_mass_x)*rel_forward_vel*rel_sideway_vel
                    
            TAM = pbm.mdl.vehicle.thruster_allocation_matrix
            pinv_TAM = pinv(TAM)
            u_des = [thrust_forwad; thrust_sideway; thrust_up; thrust_yaw]
            u_thruster_level = pinv_TAM*u_des
            u_guess[:, k] = u_thruster_level
        end
        @printf(". done\n")
        return x_guess, u_guess, p_guess
    else
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        g = pbm.mdl.env.g

        
        # Parameter guess
        p_guess = zeros(pbm.np)
        p_guess[veh.id_t] = traj.tf_guess

        # State guess
        x0 = zeros(pbm.nx)
        xf = zeros(pbm.nx)
        x0[veh.id_r] = traj.r0
        xf[veh.id_r] = traj.rf
        x0[veh.id_v] = traj.v0
        xf[veh.id_v] = traj.vf
        x_guess = zeros(pbm.nx, N)
        if traj.use_guess == 1
            xmiddle = zeros(pbm.nx)
            xmiddle[1] = x0[1]
            xmiddle[2] = xf[2]
            xmiddle[3] = 0.5*(x0[3]+xf[3])
            
            
            x_1 = straightline_interpolate(x0, xmiddle, Int(N/2))
            x_2 = straightline_interpolate(xmiddle, xf, Int(N/2))
            x_guess[:, 1:Int(N/2)] = x_1
            x_guess[:, Int(N/2)+1:N] = x_2
        elseif traj.use_guess == 2
            xmiddle = zeros(pbm.nx)
            xmiddle[2] = x0[2]
            xmiddle[1] = xf[1]
            xmiddle[3] = 0.5*(x0[3]+xf[3])
            
            x_1 = straightline_interpolate(x0, xmiddle, Int(N/2))
            x_2 = straightline_interpolate(xmiddle, xf, Int(N/2))
            x_guess[:, 1:Int(N/2)] = x_1
            x_guess[:, Int(N/2)+1:N] = x_2
        else
            x_guess = straightline_interpolate(x0, xf, N)
        end
        
        dt = 50 / (N - 1)

        # assume position path x[1,:], x[2,:], x[3,:] already chosen

        xdot = diff(x_guess[1, :]) ./ dt
        ydot = diff(x_guess[2, :]) ./ dt
        zdot = diff(x_guess[3, :]) ./ dt

        psi = zeros(N)
        u_b = zeros(N)
        v_b = zeros(N)
        w_b = zeros(N)
        r_b = zeros(N)

        for k in 1:N-1
            psi[k] = atan(ydot[k], xdot[k])   # atan2-style
            u_b[k] = sqrt(xdot[k]^2 + ydot[k]^2)
            v_b[k] = 0.0
            w_b[k] = zdot[k]
        end

        psi[end] = psi[end-1]
        u_b[end] = u_b[end-1]
        v_b[end] = v_b[end-1]
        w_b[end] = w_b[end-1]

        for k in 1:N-1
            dpsi = psi[k+1] - psi[k]
            dpsi = atan(sin(dpsi), cos(dpsi))   # wrapped difference
            r_b[k] = dpsi / dt
        end
        r_b[end] = r_b[end-1]

        x_guess[4, :] = psi
        x_guess[5, :] = u_b
        x_guess[6, :] = v_b
        x_guess[7, :] = w_b
        x_guess[8, :] = r_b


        # -------------------------
        # 6-thruster input guess
        # -------------------------
        u_guess = zeros(pbm.nu, N)
        TAM = veh.thruster_allocation_matrix
        TAM_pinv = pinv(TAM)
        hyd = pbm.mdl.hydroparams
        denx = hyd.mass - hyd.added_mass_x
        deny = hyd.mass - hyd.added_mass_y
        denz = hyd.mass - hyd.added_mass_z
        denr = hyd.inertia_z - hyd.added_mass_yaw


        # Approximate accelerations
        udot_b = zeros(N)
        vdot_b = zeros(N)
        wdot_b = zeros(N)
        rdot_b = zeros(N)

        for k in 1:(N - 1)
            udot_b[k] = (u_b[k + 1] - u_b[k]) / dt
            vdot_b[k] = (v_b[k + 1] - v_b[k]) / dt
            wdot_b[k] = (w_b[k + 1] - w_b[k]) / dt
            rdot_b[k] = (r_b[k + 1] - r_b[k]) / dt
        end
        udot_b[end] = udot_b[end - 1]
        vdot_b[end] = vdot_b[end - 1]
        wdot_b[end] = wdot_b[end - 1]
        rdot_b[end] = rdot_b[end - 1]

        # Build desired generalized wrench tau_des = [tau_x, tau_y, tau_z, tau_yaw]
        for k in 1:N
            uk = u_b[k]
            vk = v_b[k]
            wk = w_b[k]
            rk = r_b[k]

            # Current disabled in your model
            u_c = 0.0
            v_c = 0.0
            w_c = 0.0

            urx = uk - u_c
            ury = vk - v_c
            urz = wk - w_c

            coupling_x = (vk - v_c) * (hyd.mass * rk - hyd.added_mass_y * rk)
            coupling_y = (uk - u_c) * (hyd.added_mass_x * rk - hyd.mass * rk)
            coupling_yaw = (vk - v_c) * (uk - u_c) * (hyd.added_mass_y - hyd.added_mass_x)* 1.0

            drag_x = (hyd.linear_drag_x + hyd.quadratic_drag_x * abs_smooth(urx)) * urx
            drag_y = (hyd.linear_drag_y + hyd.quadratic_drag_y * abs_smooth(ury)) * ury
            drag_z = (hyd.linear_drag_z + hyd.quadratic_drag_z * abs_smooth(urz)) * urz
            drag_yaw = (hyd.linear_drag_yaw + hyd.quadratic_drag_yaw * abs_smooth(rk)) * rk

            tau_x = denx * udot_b[k] - drag_x - coupling_x 
            tau_y = deny * vdot_b[k] - drag_y - coupling_y 
            tau_z = denz * wdot_b[k] - (hyd.buoyancy - hyd.weight) - drag_z
            tau_yaw = denr * rdot_b[k] - drag_yaw - coupling_yaw

            tau_des = [tau_x, tau_y, tau_z, tau_yaw]

            # Map generalized wrench to 6 thrusters
            u_guess[:, k] = TAM_pinv * tau_des

            # Clip to actuator bounds
            u_guess[:, k] = clamp.(u_guess[:, k], veh.u_min, veh.u_max)
        end

        return x_guess, u_guess, p_guess
    end
end

function set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(pbm, auv_initial_guess)
    return nothing
end

function set_cost!(pbm::TrajectoryProblem)::Nothing

    problem_set_terminal_cost!(
        pbm,
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            tdil = p[veh.id_t]
            tdil_max = traj.tf_max
            γ = traj.γ
            return γ * (tdil / tdil_max)^2
        end,
    )

    problem_set_running_cost!(
        pbm,
        :scvx,
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            γ = pbm.mdl.traj.γ
            P = get_power_from_thrust(u)
            P_max = get_power_from_thrust(ones(size(u))*2.5)
            yaw_rate_max = 0.3
            yaw_rate = x[8]
            #@show typeof(P)
            return γ*P/P_max #+ yaw_rate^2/yaw_rate_max
        end,
    )

    return nothing
end


function abs_smooth(z; ϵ = 1e-3)    
    return sqrt(z*z + oftype(z, 1e-3)^2)
end

"""
    get_current(state; yawidx=4, shear=0.2, width=0.5)

    Returns (u_c, v_c, w_c)

"""
function get_current(state::AbstractVector;
    xidx::Int=1,
    yidx::Int=2,
    yawidx::Int=4,
    vmax::Real=-0.5,
    offset::Real=4.0,      # max speed occurs at |d| = 4 m
    sigma::Real=1.5        # controls band thickness
)
    x = state[xidx]
    y = state[yidx]
    ψ = state[yawidx]

    # Line: 3.56x - y - 27.0312 = 0
    a = 3.56
    b = -1.0
    c = -27.0312

    # Signed perpendicular distance to the line
    d = (a*x + b*y + c) / sqrt(a^2 + b^2)

    # Unit vector parallel to the line, in +x,+y direction
    tx = 1.0
    ty = 3.56
    tnorm = sqrt(tx^2 + ty^2)
    tx /= tnorm
    ty /= tnorm

    # Smooth magnitude, max at |d| = offset
    speed = vmax * exp(-((abs(d) - offset)^2) / sigma^2)

    # Inertial/world-frame current
    u_i = speed * tx
    v_i = speed * ty

    sψ = sin(ψ)
    cψ = cos(ψ)

    # WF -> body
    u_c =  cψ*u_i + sψ*v_i
    v_c = -sψ*u_i + cψ*v_i
    w_c = zero(eltype(state))

    return u_c, v_c, w_c
end


function get_power_from_thrust(u_i::AbstractVector; real::Bool=false)

    g0 = 9.80665
    sumP = zero(eltype(u_i))

    # During optimization use simple quadratic (with minimum at (0,0))
    if !real
        #coeff = (17.37552416, 0.0, 0.0)
        coeff = (48.58943943, 0.0, 0.0)
        for ui in u_i
            τ = ui / g0
            
            pi = evalpoly(τ, reverse(coeff))
            
            sumP += pi
        end
    # During evaluation use more realistic approximation (with minimum at (0,0))
    # Note that the thrust to power ratio is not symmetric
    else
        coeff_pos = (0.05482628,  -0.79935459,   4.83666659, -16.1514991,   39.42642134, 0.0, 0.0)
        coeff_neg = (0.41679474,  4.37816223, 18.16230606, 39.45115415, 64.26514808,  0.0, 0.0)

        for ui in u_i
            τ = ui / g0
            if τ >= 0
                pi = evalpoly(τ, reverse(coeff_pos))
            else
                pi = evalpoly(τ, reverse(coeff_neg))
            end
            sumP += pi
        end
    end
    return sumP
end

"""
    dynamics(t, k, x, u, p, pbm[; no_hydro_torques])

AUV vehicle dynamics.

Args:
- `t`: the current time (normalized).
- `k`: the current discrete-time node.
- `x`: the current state vector.
- `u`: the current input vector.
- `p`: the parameter vector.
- `pbm`: the AUV landing flip problem description.
- `no_hydro_torques`: (optional) whether to omit torques generated by coriolis and
  drag.

Returns:
- `f`: the time derivative of the state vector.
"""
function dynamics(
    t::Real,
    k::Int,
    x::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
    pbm::TrajectoryProblem;
    no_hydro_torques::Bool = false,
)

    # Parameters
    veh = pbm.mdl.vehicle
    hyd = pbm.mdl.hydroparams
    traj = pbm.mdl.traj

    TAM = pbm.mdl.vehicle.thruster_allocation_matrix
    
    # Current (x, u, p) values
    tdil = p[veh.id_t]

    T = promote_type(eltype(x), eltype(u), eltype(p))

    # Promote type to support Duals (x or u can be Dual during AD)
    f = zeros(T, pbm.nx)

    @views pos = x[veh.id_r]          # 4
    @views v   = x[veh.id_v]          # 4
    @views ut  = u[veh.id_u]          # 6
    
    ψ = pos[4]
    cψ = cos(ψ)
    sψ = sin(ψ)
  
    # thrusters -> body forces
    uxyzyaw = zeros(T, 4)
    mul!(uxyzyaw, TAM, ut)

    # currents
    u_c = zero(T); v_c = zero(T); w_c = zero(T)
    u_c, v_c, w_c = get_current(pos)

    # helpers
    denx = hyd.mass - hyd.added_mass_x
    deny = hyd.mass - hyd.added_mass_y
    denz = hyd.mass - hyd.added_mass_z
    denr = hyd.inertia_z - hyd.added_mass_yaw
    urx = v[1] - u_c
    ury = v[2] - v_c
    urz = v[3] - w_c

 
    yaw_moment = ((ury)*(urx)*(hyd.added_mass_y - hyd.added_mass_x)* 1.0
        + (hyd.linear_drag_yaw + hyd.quadratic_drag_yaw*abs_smooth(v[4]))*v[4])
    #yaw_moment = ((hyd.linear_drag_yaw + hyd.quadratic_drag_yaw*abs_smooth(v[4]))*v[4])

    # The dynamics
    f[1] = cψ*v[1] - sψ*v[2]
    f[2] = sψ*v[1] + cψ*v[2]
    f[3] = v[3]
    f[4] = v[4]
    f[5] = (uxyzyaw[1]
            + (ury)*(hyd.mass*v[4] - hyd.added_mass_y*v[4])
            + (hyd.linear_drag_x + hyd.quadratic_drag_x*abs_smooth(urx))*urx) / denx

    f[6] = (uxyzyaw[2]
            + (urx)*(hyd.added_mass_x*v[4] - hyd.mass*v[4])
            + (hyd.linear_drag_y + hyd.quadratic_drag_y*abs_smooth(ury))*ury) / deny

    f[7] = (uxyzyaw[3] + hyd.buoyancy - hyd.weight
            + (hyd.linear_drag_z + hyd.quadratic_drag_z*abs_smooth(urz))*urz) / denz

    f[8] = (uxyzyaw[4]
            + yaw_moment) / denr

    # Scale for time
    f *= tdil
    return f
end


function dynamics_symbolic(x, u, p, pbm)
    return dynamics(0.0, 1, x, u, p, pbm)
end

function set_dynamics!(pbm::TrajectoryProblem)::Nothing

    Symbolics.@variables xsym[1:pbm.nx]
    Symbolics.@variables usym[1:pbm.nu]
    Symbolics.@variables psym[1:pbm.np]

    dyn_symb = dynamics_symbolic(xsym, usym, psym, pbm)

    A_symb = Symbolics.jacobian(dyn_symb, xsym)
    B_symb = Symbolics.jacobian(dyn_symb, usym)
    F_symb = Symbolics.jacobian(dyn_symb, psym)

    vars = [xsym...; usym...; psym]

    f_oop, f_ip = build_function(
        dyn_symb,
        xsym, usym, psym;
        expression = Val(false),
        parallel = Symbolics.SerialForm(),
    )

    A_oop, A_ip = build_function(
        A_symb,
        xsym, usym, psym;
        expression = Val(false),
        parallel = Symbolics.SerialForm(),
    )

    B_oop, B_ip = build_function(
        B_symb,
        xsym, usym, psym;
        expression = Val(false),
        parallel = Symbolics.SerialForm(),
    )

    F_oop, F_ip = build_function(
        F_symb,
        xsym, usym, psym;
        expression = Val(false),
        parallel = Symbolics.SerialForm(),
    )

    f_cb = function(t, k, x, u, p, pbm)
        out = zeros(eltype(x), pbm.nx)
        f_ip(out, x, u, p)
        return out
    end

    A_cb = function(t, k, x, u, p, pbm)
        out = zeros(eltype(x), pbm.nx, pbm.nx)
        A_ip(out, x, u, p)
        return out
    end

    B_cb = function(t, k, x, u, p, pbm)
        out = zeros(eltype(x), pbm.nx, pbm.nu)
        B_ip(out, x, u, p)
        return out
    end

    F_cb = function(t, k, x, u, p, pbm)
        out = zeros(eltype(x), pbm.nx, pbm.np)
        F_ip(out, x, u, p)
        return out
    end

    problem_set_dynamics!(pbm, f_cb, A_cb, B_cb, F_cb)
        """A_auto = (t,k,x,u,p,pbm) -> ForwardDiff.jacobian(xx -> dynamics(t,k,xx,u,p,pbm), x)
    B_auto = (t,k,x,u,p,pbm) -> ForwardDiff.jacobian(uu -> dynamics(t,k,x,uu,p,pbm), u)
    F_auto = (t,k,x,u,p,pbm) -> ForwardDiff.jacobian(pp -> dynamics(t,k,x,u,pp,pbm), p)"""
    return nothing
end



function set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the state
    problem_set_X!(
        pbm,
        (t, k, x, p, pbm, ocp) -> begin
            traj = pbm.mdl.traj
            env = pbm.mdl.env
            veh = pbm.mdl.vehicle
            u_robot = x[veh.id_v][1]
            v_robot = x[veh.id_v][2]
            z_pos = x[veh.id_r][3]
            tdil = p[veh.id_t]
            x_pos = x[veh.id_r][1]
            y_pos = x[veh.id_r][2]

            @add_constraint(ocp, NONPOS, "forward_motion", (u_robot,), begin
                local u = arg[1]
                -u
            end)

            @add_constraint(ocp, NONPOS, "z_underwater", (z_pos,), begin
                local z = arg[1]
                z
            end)
            
            @add_constraint(ocp, NONPOS, "sideways_vel_small_neg", (u_robot, v_robot), begin
                    local u, v = arg
                    -v - 0.2*u      # Velocity v >= -0.2*u 
                end)

            @add_constraint(ocp, NONPOS, "sideways_vel_small_pos", (u_robot, v_robot), begin
                    local u, v = arg
                    v - 0.2*u    # Velocity v <= 0.2*u 
                end)

            @add_constraint(ocp, NONPOS, "sideways_vel_small_pos", (x_pos, y_pos), begin
                    local x, y = arg
                    3.56*x - y - 27.0312    # Harbour Wall constraint 3.56*x - y <= 27.0312 
                end)

            # @add_constraint(ocp, NONPOS, "max_time", (tdil,), begin
            #         local tdil_max = arg[1]
            #         tdil_max - traj.tf_max
            #     end)

            # @add_constraint(ocp, NONPOS, "min_time", (tdil,), begin
            #         local tdil_min = arg[1]
            #         traj.tf_min - tdil_min
            #     end)
        end,
    )

    # Convex path constraints on the input
    # problem_set_U!(
    #     pbm,
    #     (t, k, u, p, pbm, ocp) -> begin
    #         veh = pbm.mdl.vehicle
    #         traj = pbm.mdl.traj

    #         T = u[veh.id_u]
    #         T_min = veh.u_min
    #         T_max = veh.u_max

    #         # Thruster 1
    #         @add_constraint(ocp, NONPOS, "max_thrust_1", (T[1],), begin
    #             local T = arg[1]
    #             T - T_max
    #         end)
    #         @add_constraint(ocp, NONPOS, "min_thrust_1", (T[1],), begin
    #             local T = arg[1]
    #             T_min - T
    #         end)
    #         # Thruster 2
    #         @add_constraint(ocp, NONPOS, "max_thrust_2", (T[2],), begin
    #             local T = arg[1]
    #             T - T_max
    #         end)
    #         @add_constraint(ocp, NONPOS, "min_thrust_2", (T[2],), begin
    #             local T = arg[1]
    #             T_min - T
    #         end)
    #         # Thruster 3
    #         @add_constraint(ocp, NONPOS, "max_thrust_3", (T[3],), begin
    #             local T = arg[1]
    #             T - T_max
    #         end)
    #         @add_constraint(ocp, NONPOS, "min_thrust_3", (T[3],), begin
    #             local T = arg[1]
    #             T_min - T
    #         end)
    #         # Thruster 4
    #         @add_constraint(ocp, NONPOS, "max_thrust_4", (T[4],), begin
    #             local T = arg[1]
    #             T - T_max
    #         end)
    #         @add_constraint(ocp, NONPOS, "min_thrust_4", (T[4],), begin
    #             local T = arg[1]
    #             T_min - T
    #         end)
    #         # Thruster 5
    #         @add_constraint(ocp, NONPOS, "max_thrust_5", (T[5],), begin
    #             local T = arg[1]
    #             T - T_max
    #         end)
    #         @add_constraint(ocp, NONPOS, "min_thrust_5", (T[5],), begin
    #             local T = arg[1]
    #             T_min - T
    #         end)
    #         # Thruster 6
    #         @add_constraint(ocp, NONPOS, "max_thrust_6", (T[6],), begin
    #             local T = arg[1]
    #             T - T_max
    #         end)
    #         @add_constraint(ocp, NONPOS, "min_thrust_6", (T[6],), begin
    #             local T = arg[1]
    #             T_min - T
    #         end)
    #     end,
    # )

    return nothing
end

function set_nonconvex_constraints!(pbm::TrajectoryProblem, algo::Symbol)::Nothing
    # So far no obstacles -> no nonconvex constraints
    # Constraint s
    _q__s = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        s = zeros(env.n_obs)
        for i = 1:env.n_obs
            E = env.obs[i]
            r = x[veh.id_r]
            s[i] = 1 - E(r)
        end
        return s
    end

    # Jacobian ds/dx
    _q__C = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        C = zeros(env.n_obs, pbm.nx)
        for i = 1:env.n_obs
            E = env.obs[i]
            r = x[veh.id_r]
            C[i, veh.id_r] = -∇(E, r)
        end
        return C
    end

    if algo == :scvx
        problem_set_s!(pbm, algo, _q__s, _q__C)
    else
        _q___s = (t, k, x, p, pbm) -> _q__s(t, k, x, nothing, p, pbm)
        _q___C = (t, k, x, p, pbm) -> _q__C(t, k, x, nothing, p, pbm)
        problem_set_s!(pbm, algo, _q___s, _q___C)
    end

end

function set_bcs!(pbm::TrajectoryProblem)::Nothing

    # Initial conditions
    problem_set_bc!(
        pbm, :ic,
        # Constraint g
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        rhs = zeros(pbm.nx)
        rhs[veh.id_r] = traj.r0
        rhs[veh.id_v] = traj.v0
        g = x-rhs
        return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
        H = I(pbm.nx)
        return H
        end,
        # Jacobian dg/dp
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        K = zeros(pbm.nx, pbm.np)
        return K
        end)

    # Terminal conditions
    problem_set_bc!(
        pbm, :tc,
        # Constraint g
        (x, p, pbm) -> begin
            veh  = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            # Free final yaw
            free_idx = veh.id_r[4]
            # Constrain every terminal state except yaw
            keep_idx = setdiff(1:pbm.nx, [free_idx])
            rhs = zeros(pbm.nx)
            rhs[veh.id_r] = traj.rf
            rhs[veh.id_v] = traj.vf
            g = x[keep_idx] - rhs[keep_idx]
            return g
        end,

        # Jacobian dg/dx
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            free_idx = veh.id_r[4]
            keep_idx = setdiff(1:pbm.nx, [free_idx])
            H = Matrix(I, pbm.nx, pbm.nx)[keep_idx, :]
            return H
        end,

        # Jacobian dg/dp
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            free_idx = veh.id_r[4]
            keep_idx = setdiff(1:pbm.nx, [free_idx])
            K = zeros(length(keep_idx), pbm.np)
            return K
        end
    )

    return nothing
end
