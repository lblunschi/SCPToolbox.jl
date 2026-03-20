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
    min_x = minimum([x0[1],xf[1]])
    max_x = maximum([x0[1],xf[1]])
    min_y = minimum([x0[2],xf[2]])
    max_y = maximum([x0[2],xf[2]])
    min_z = minimum([x0[3],xf[3]])
    max_z = maximum([x0[3],xf[3]])
    # Position
    advise!(pbm, :state, veh.id_r[1], (min_x, max_x))
    advise!(pbm, :state, veh.id_r[2], (min_y, max_y))
    advise!(pbm, :state, veh.id_r[3], (min_z, max_z))
    advise!(pbm, :state, veh.id_r[4], (-3.1415/4, 3.1415/4)) # TODO check how to scale yaw
    # Velocity
    advise!(pbm, :state, veh.id_v[1], (veh.v_min_u, veh.v_max))
    advise!(pbm, :state, veh.id_v[2:3], (veh.v_min, veh.v_max))
    advise!(pbm, :state, veh.id_v[4], (-0.2, 0.2)) # TODO check how to scale yaw rate
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
            #@show typeof(P)
            return γ*P/P_max
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
    yawidx::Int=4,
    shear::Real=0.2,
    width::Real=0.5
)
    x = state[1]
    ψ = state[yawidx]

    # inertial shear in +/−y (smooth) WF
    v_i = -shear * tanh(x / width)

    sψ = sin(ψ)
    cψ = cos(ψ)

    # WF -> body (u_i = 0)
    u_c = sψ * v_i
    v_c = cψ * v_i
    w_c = zero(eltype(state))

    #return 0, 0, 0
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

 
    yaw_moment = ((ury)*(urx)*(hyd.added_mass_y - hyd.added_mass_x)
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

            @add_constraint(ocp, NONPOS, "max_time", (tdil,), begin
                    local tdial = arg[1]
                    tdial - traj.tf_max
                end)

            @add_constraint(ocp, NONPOS, "min_time", (tdil,), begin
                    local tdil = arg[1]
                    traj.tf_min - tdil
                end)
        end,
    )

    # Convex path constraints on the input
    problem_set_U!(
        pbm,
        (t, k, u, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            T = u[veh.id_u]
            T_min = veh.u_min
            T_max = veh.u_max

            # Thruster 1
            @add_constraint(ocp, NONPOS, "max_thrust_1", (T[1],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_1", (T[1],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 2
            @add_constraint(ocp, NONPOS, "max_thrust_2", (T[2],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_2", (T[2],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 3
            @add_constraint(ocp, NONPOS, "max_thrust_3", (T[3],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_3", (T[3],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 4
            @add_constraint(ocp, NONPOS, "max_thrust_4", (T[4],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_4", (T[4],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 5
            @add_constraint(ocp, NONPOS, "max_thrust_5", (T[5],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_5", (T[5],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 6
            @add_constraint(ocp, NONPOS, "max_thrust_6", (T[6],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_6", (T[6],), begin
                local T = arg[1]
                T_min - T
            end)
        end,
    )

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
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        rhs = zeros(pbm.nx)
        rhs[veh.id_r] = traj.rf
        rhs[veh.id_v] = traj.vf
        g = x-rhs
        return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        H = I(pbm.nx)
        return H
        end,
        # Jacobian dg/dp
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        K = zeros(pbm.nx, pbm.np)
        return K
        end)

    return nothing
end
