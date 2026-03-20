#= AUVSimplified landing flip maneuver problem definition.

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

# ..:: Methods ::..

function define_problem_simplified!(pbm::TrajectoryProblem, algo::Symbol)::Nothing
    set_dims_simp!(pbm)
    set_scale_simp!(pbm)
    set_dynamics_simp!(pbm)
    set_convex_constraints_simp!(pbm)
    set_nonconvex_constraints_simp!(pbm, algo)
    set_bcs_simp!(pbm)
    set_cost_simp!(pbm)

    set_guess_simp!(pbm)

    return nothing
end

function set_dims_simp!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 9, 4, 1) # nx, nu, np

    return nothing
end

function set_scale_simp!(pbm::TrajectoryProblem)::Nothing

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
    advise!(pbm, :state, veh.id_r[4], (-3.1415/4, 3.1415/4))
    # Velocity
    advise!(pbm, :state, veh.id_v[1], (veh.v_min_u, veh.v_max))
    advise!(pbm, :state, veh.id_v[2:3], (veh.v_min, veh.v_max))
    advise!(pbm, :state, veh.id_v[4], (-3.1415/10, 3.1415/10))
    # Inputs
    advise!(pbm, :input, veh.id_u, (veh.u_min, veh.u_max))

    return nothing
end

""" Compute the initial trajectory guess.

This uses a simple bang-bang control strategy for the flip maneuver. Once
AUVSimplified is upright, convex optimization is used to find the terminal descent
trajectory by approximatin AUVSimplified as a double-integrator (no attitude, no
aerodynamics).

Args:
* `N`: the number of discrete-time grid nodes.
* `pbm`: the trajectory problem structure.

Returns:
* `x_guess`: the state trajectory initial guess.
* `u_guess`: the input trajectory initial guess.
* `p_guess`: the parameter vector initial guess.
"""
function auv_simplified_initial_guess(
    N::Int,
    pbm::TrajectoryProblem,
)::Tuple{RealMatrix,RealMatrix,RealVector}

    @printf("Computing initial guess .")

    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj
    g = pbm.mdl.env.g
    hyd = pbm.mdl.hydroparams
    t_dial = traj.tf
    p_guess = zeros(1)
    p_guess[1] = traj.tf
    # State guess
    x0 = zeros(pbm.nx)
    xf = zeros(pbm.nx)
    x0[veh.id_r] = traj.r0
    xf[veh.id_r] = traj.rf
    x0[veh.id_v] = traj.v0
    xf[veh.id_v] = traj.vf
    x0[veh.id_e] = 0.0
    xf[veh.id_e] = 200

    x_guess = straightline_interpolate(x0, xf, N)
    dist = norm(xf[1:2]-x0[1:2])
    forward_vel = dist/t_dial
    dist_up = xf[3]-x0[3]
    up_vel = dist_up/t_dial
    for t in 1:N
        x_guess[5, t] = forward_vel
        x_guess[7, t] = up_vel
    end
    

    thrust_forwad = -(hyd.linear_drag_x + hyd.quadratic_drag_x*abs(forward_vel))*forward_vel

    thrust_sideway = 0.0

    thrust_up = (- hyd.buoyancy + hyd.weight - (hyd.linear_drag_z + hyd.quadratic_drag_z*abs(up_vel))*up_vel)

    thrust_yaw = 0.0
            
    TAM = pbm.mdl.vehicle.thruster_allocation_matrix
    pinv_TAM = pinv(TAM)
    u_des = [thrust_forwad; thrust_sideway; thrust_up; thrust_yaw]

    u_guess = straightline_interpolate(u_des, u_des, N)
    @printf(". done\n")
    return x_guess, u_guess, p_guess
end

function set_guess_simp!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(pbm, auv_simplified_initial_guess)

    return nothing
end

function set_cost_simp!(pbm::TrajectoryProblem)::Nothing

    problem_set_running_cost!(
            pbm,
            :scvx,
            (t, k, x, u, p, pbm) -> begin
                
                return sum(u.^2)
            end,
        )

    return nothing
end

function abs_smooth_simp(z; ϵ = 1e-3)    
    return sqrt(z*z + oftype(z, 1e-3)^2)
end

"""
    get_current(state; yawidx=4, shear=0.2, width=0.5)

    Returns (u_c, v_c, w_c)

"""
function get_current_simp(state::AbstractVector;
    yawidx::Int=4,
    shear::Real=0.3,
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

    return 0, 0, 0
    #return u_c, v_c, w_c
end


function get_power_from_thrust_simp(u_i::AbstractVector)
    # Both fits ensure first order continuity (f(0) = f'(0) = 0.0)
    # Fit for τ >= 0
    coeff_pos = (0.05482628,  -0.79935459,   4.83666659, -16.1514991,   39.42642134, 0.0, 0.0)
    coeff_pos = (20.0, 0.0, 0.0)
    # Fit for τ < 0
    coeff_neg = (0.41679474,  4.37816223, 18.16230606, 39.45115415, 64.26514808,  0.0, 0.0)
    coeff_neg = (24.35312162, 0.0, 0.0)

    g0 = 9.80665
    sumP = zero(eltype(u_i))

    # TEST: to see if evalpoly is causing the issues
    for ui in u_i
        τ = ui / g0
        
        pi = evalpoly(τ, reverse(coeff_pos))
        
        sumP += pi
    end
    """for ui in u_i
        τ = ui / g0
        if τ >= 0
            pi = evalpoly(τ, reverse(coeff_pos))
        else
            pi = evalpoly(τ, reverse(coeff_neg))
        end
        sumP += pi
    end"""
    return sumP
end

"""
    dynamics(t, k, x, u, p, pbm[; no_hydro_torques])

AUVSimplified vehicle dynamics.

Args:
- `t`: the current time (normalized).
- `k`: the current discrete-time node.
- `x`: the current state vector.
- `u`: the current input vector.
- `p`: the parameter vector.
- `pbm`: the AUVSimplified landing flip problem description.
- `no_hydro_torques`: (optional) whether to omit torques generated by coriolis and
  drag.

Returns:
- `f`: the time derivative of the state vector.
"""
function dynamics_simp(
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
    tdil = traj.tf
    # Promote type to support Duals (x or u can be Dual during AD)
    T = promote_type(eltype(x), eltype(u), eltype(p))
    f = zeros(T, pbm.nx)
    @views pos = x[veh.id_r]          # 4
    @views v   = x[veh.id_v]          # 4
    @views ut  = u[veh.id_u]          # 4
    
    ψ = pos[4]
    cψ = cos(ψ)
    sψ = sin(ψ)
  
    # thrusters -> body forces (4×6 * 6)

    power_from_thrust = get_power_from_thrust_simp(ut)
    # currents
    u_c = zero(T); v_c = zero(T); w_c = zero(T)
    u_c, v_c, w_c = get_current_simp(pos)
    # helpers

    denx = hyd.mass - hyd.added_mass_x
    deny = hyd.mass - hyd.added_mass_y
    denz = hyd.mass - hyd.added_mass_z
    denr = hyd.inertia_z - hyd.added_mass_yaw
    urx = v[1] - u_c
    ury = v[2] - v_c
    urz = v[3] - w_c

    yaw_moment = 0.0
    if !no_hydro_torques
        yaw_moment = ((v[2]-v_c)*(v[1]-u_c)*(hyd.added_mass_y - hyd.added_mass_x)
            + (hyd.linear_drag_yaw + hyd.quadratic_drag_yaw*abs_smooth_simp(v[4]))*v[4])
    else
        yaw_moment = 0.0
    end
    yaw_moment = ((v[2]-v_c)*(v[1]-u_c)*(hyd.added_mass_y - hyd.added_mass_x)
            + (hyd.linear_drag_yaw + hyd.quadratic_drag_yaw*abs_smooth_simp(v[4]))*v[4])
    # The dynamics
    f[1] = cψ*v[1] - sψ*v[2]
    f[2] = sψ*v[1] + cψ*v[2]
    f[3] = v[3]
    f[4] = v[4]
    f[5] = (ut[1]
            + (v[2]-v_c)*(hyd.mass*v[4] - hyd.added_mass_y*v[4])
            + (hyd.linear_drag_x + hyd.quadratic_drag_x*abs_smooth_simp(urx))*urx) / denx

    f[6] = (ut[2]
            + (v[1]-u_c)*(hyd.added_mass_x*v[4] - hyd.mass*v[4])
            + (hyd.linear_drag_y + hyd.quadratic_drag_y*abs_smooth_simp(ury))*ury) / deny

    f[7] = (ut[3] + hyd.buoyancy - hyd.weight
            + (hyd.linear_drag_z + hyd.quadratic_drag_z*abs_smooth_simp(urz))*urz) / denz

    f[8] = (ut[4]
            + yaw_moment) / denr
    f[9] = power_from_thrust 

    # Scale for time
    f *= tdil
    return f
end

function set_dynamics_simp!(pbm::TrajectoryProblem)::Nothing
    A_auto = (t,k,x,u,p,pbm) -> ForwardDiff.jacobian(xx -> dynamics_simp(t,k,xx,u,p,pbm), x)
    B_auto = (t,k,x,u,p,pbm) -> ForwardDiff.jacobian(uu -> dynamics_simp(t,k,x,uu,p,pbm), u)
    F_auto = (t,k,x,u,p,pbm) -> ForwardDiff.jacobian(pp -> dynamics_simp(t,k,x,u,pp,pbm), p)

    problem_set_dynamics!(
        pbm,
        # Dynamics f
        (t, k, x, u, p, pbm) -> begin
            f = dynamics_simp(t, k, x, u, p, pbm)
            return f
        end,
        # Jacobian df/dx
        A_auto,
        # Jacobian df/du
        B_auto,
        # Jacobian df/dp
        F_auto,
    )

    return nothing
end

function set_convex_constraints_simp!(pbm::TrajectoryProblem)::Nothing

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
            energy = x[veh.id_e]

            @add_constraint(ocp, NONPOS, "forward_motion", (u_robot,), begin
                local u = arg[1]
                -u
            end)

            @add_constraint(ocp, NONPOS, "z_underwater", (z_pos,), begin
                local z = arg[1]
                z
            end)
            
            @add_constraint(ocp, NONPOS, "energy_positive", (energy,), begin
                local E = arg[1]
                -E
            end)

            @add_constraint(ocp, NONPOS, "sideways_vell_small_neg", (u_robot, v_robot), begin
                    local u, v = arg
                    -v - 0.2*u      # Velocity v >= -0.2*u 
                end)

            @add_constraint(ocp, NONPOS, "sideways_vell_small_pos", (u_robot, v_robot), begin
                    local u, v = arg
                    v - 0.2*u    # Velocity v <= 0.2*u 
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
            @add_constraint(ocp, NONPOS, "max_thrust_x", (T[1],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_x", (T[1],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 2
            @add_constraint(ocp, NONPOS, "max_thrust_y", (T[2],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_y", (T[2],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 3
            @add_constraint(ocp, NONPOS, "max_thrust_z", (T[3],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_z", (T[3],), begin
                local T = arg[1]
                T_min - T
            end)
            # Thruster 4
            @add_constraint(ocp, NONPOS, "max_thrust_yaw", (T[4],), begin
                local T = arg[1]
                T - T_max
            end)
            @add_constraint(ocp, NONPOS, "min_thrust_yaw", (T[4],), begin
                local T = arg[1]
                T_min - T
            end)
        end,
    )

    return nothing
end

function set_nonconvex_constraints_simp!(pbm::TrajectoryProblem, algo::Symbol)::Nothing
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

function set_bcs_simp!(pbm::TrajectoryProblem)::Nothing

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
        rhs = zeros(pbm.nx-1)
        rhs[veh.id_r] = traj.rf
        rhs[veh.id_v] = traj.vf
        g = x[1:(veh.id_e-1)]-rhs
        return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        H = zeros(pbm.nx - 1, pbm.nx)
        H[:, 1:pbm.nx-1] .= I(pbm.nx - 1)
        return H
        end,
        # Jacobian dg/dp
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        K = zeros(pbm.nx-1, pbm.np)
        return K
        end)

    return nothing
end
