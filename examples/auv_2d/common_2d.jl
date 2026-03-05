#= AUV obstacle avoidance example, common code.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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
using ForwardDiff

include("../../models/auv.jl")
include("../../core/problem.jl")
include("../../utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function define_problem!(pbm::TrajectoryProblem,
                         algo::T_Symbol)::Nothing
    _common__set_dims!(pbm)
    _common__set_scale!(pbm)
    _common__set_cost!(pbm, algo)
    _common__set_dynamics!(pbm)
    _common__set_convex_constraints!(pbm)
    _common__set_nonconvex_constraints!(pbm, algo)
    _common__set_bcs!(pbm)

    _common__set_guess!(pbm)

    return nothing
end

function _common__set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 8, 4, 1) # nx, nu, np

    return nothing
end

function _common__set_scale!(pbm::TrajectoryProblem)::Nothing
    """Set problem scaling for better numerical conditioning."""
    # Adjust the scaling
    # New variable bounded in [0, 1]
    # Original variable = scale * new variable + offset
    # TODO see if this makes sense. Computation is just setting adj to max..
    mdl = pbm.mdl

    # Set time scaling
    tdil_min = mdl.traj.tf_min
    tdil_max = mdl.traj.tf_max
    tdil_max_adj = tdil_min+1.0*(tdil_max-tdil_min)
   
    problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t,
                          (tdil_min, tdil_max_adj))

    # Set input scaling
    u_min = mdl.vehicle.u_min            
    u_max = mdl.vehicle.u_max
    u_max_adj = u_min .+ 1.0*(u_max .- u_min)          
    problem_advise_scale!(pbm, :input, mdl.vehicle.id_u,
                          (u_min, u_max_adj))

    # Set velocity scaling
    v_min = mdl.vehicle.v_min            
    v_max = mdl.vehicle.v_max
    v_max_adj = v_min .+ 1.0*(v_max .- v_min)     
    index_without_u = mdl.vehicle.id_v[2:3] # only scale sway and heave velocity, not forward velocity     
    problem_advise_scale!(pbm, :state, index_without_u,
                          (v_min, v_max_adj))

    # Set u scale
    v_min_u = mdl.vehicle.v_min_u
    v_max_u = mdl.vehicle.v_max
    v_max_u_adj = v_min_u + 1.0*(v_max_u - v_min_u)
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_u[1],
                          (v_min_u, v_max_u_adj))

    return nothing
end

function _common__set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(
        pbm, (N, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        g = pbm.mdl.env.g


        # Parameter guess
        p = zeros(pbm.np)
        p[veh.id_t] = 0.5*(traj.tf_min+traj.tf_max)

        # State guess
        x0 = zeros(pbm.nx)
        xf = zeros(pbm.nx)
        x0[veh.id_r] = traj.r0
        xf[veh.id_r] = traj.rf
        x0[veh.id_v] = traj.v0
        xf[veh.id_v] = traj.vf
        xmiddle = zeros(pbm.nx)
        xmiddle[2] = x0[2]
        xmiddle[1] = xf[1]
        xmiddle[3] = 0.5*(x0[3]+xf[3])
        x = zeros(pbm.nx, N)
        x_1 = straightline_interpolate(x0, xmiddle, Int(N/2))
        x_2 = straightline_interpolate(xmiddle, xf, Int(N/2))
        x[:, 1:Int(N/2)] = x_1
        x[:, Int(N/2)+1:N] = x_2
        x = straightline_interpolate(x0, xf, N)
        for t in 1:size(x,2)
            x[5, t] = 0.5 
        end


        hover = zeros(pbm.nu)
        hover[3] = pbm.mdl.hydroparams.weight - pbm.mdl.hydroparams.buoyancy
        u = straightline_interpolate(hover, hover, N)

        return x, u, p
        end)

    return nothing
end

function _common__set_cost!(pbm::TrajectoryProblem,
                            algo::T_Symbol)::Nothing

    problem_set_terminal_cost!(
        pbm, (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        tdil = p[veh.id_t]
        tdil_max = traj.tf_max
        γ = traj.γ
        return γ*(tdil/tdil_max)^2*10 
        end)

    # Running cost
    if algo==:scvx
        problem_set_running_cost!(
            pbm, algo,
            (x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj

            hover = norm(env.g)
            γ = traj.γ
            cost_u = sum(u[i]^2 for i in 1:4)
            #cost_v = (x[6]/0.05)^2 # penalize sway velocity wrt to u vel (we want to move predominantly forward)
            return cost_u
            end)
    else
        problem_set_running_cost!(
            pbm, algo,
            # Input quadratic penalty S
            (p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj
            hover = norm(env.g)
            γ = traj.γ
            S = zeros(pbm.nu, pbm.nu)
            return S
            end)
    end


    return nothing
end

function abs_smooth(x; ϵ = 1e-3)
    ϵT = oftype(x, ϵ)
    return sqrt(x^2 + ϵT^2)
end

"""
    get_current_body_smooth_and_partials(state; yawidx=4, shear=0.2, width=0.5)

Returns (u_c, v_c, w_c, du_dx, dv_dx, du_dψ, dv_dψ)

All derivatives are w.r.t. inertial x and yaw ψ (treating other states constant).
"""
function get_current_body_smooth_and_partials(state::AbstractVector;
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

    # dv_i/dx = -shear/width * sech²(x/width)
    a = x / width
    sech2 = 1 / (cosh(a)^2)
    dv_i_dx = -(shear / width) * sech2

    # partials
    du_dx = sψ * dv_i_dx
    dv_dx = cψ * dv_i_dx

    du_dψ =  cψ * v_i
    dv_dψ = -sψ * v_i
    #return 0, 0, 0, 0, 0, 0, 0
    return u_c, v_c, w_c, du_dx, dv_dx, du_dψ, dv_dψ
end

function _common__set_dynamics!(pbm::TrajectoryProblem)::Nothing
    # Current velocity (assumed zero for now, TODO implement current models)
    # Irrotational current, so no yaw component
    u_c = 0.0
    v_c = 0.0
    w_c = 0.0
    f_impl = (t, k, x, u, p, pbm) -> begin
            u_c, v_c, w_c, du_dx, dv_dx, du_dψ, dv_dψ = get_current_body_smooth_and_partials(x)
            veh = pbm.mdl.vehicle
            pos = x[veh.id_r]
            v   = x[veh.id_v]
            ut  = u[veh.id_u]
            tdil = p[veh.id_t]

            T = eltype(x)
            f = zeros(T, pbm.nx)   # <-- change

            f[1] = cos(pos[4])*v[1]-sin(pos[4])*v[2]
            f[2] = sin(pos[4])*v[1]+cos(pos[4])*v[2]
            f[3] = v[3]
            f[4] = v[4]

            f[5] = (1/(hydprms.mass-hydprms.added_mass_x)) * (
                    ut[1]
                    + (v[2] - v_c)*(hydprms.mass*v[4] - hydprms.added_mass_y*v[4])
                    + (hydprms.linear_drag_x + hydprms.quadratic_drag_x * abs_smooth(v[1] - u_c; ϵ=1e-3)) * (v[1] - u_c)
                )

            f[6] = (1/(hydprms.mass-hydprms.added_mass_y)) * (
                    ut[2]
                    + (v[1] - u_c)*(hydprms.added_mass_x*v[4] - hydprms.mass*v[4])
                    + (hydprms.linear_drag_y + hydprms.quadratic_drag_y * abs_smooth(v[2] - v_c; ϵ=1e-3)) * (v[2] - v_c)
                )

            f[7] = (1/(hydprms.mass-hydprms.added_mass_z)) * (
                    ut[3] + hydprms.buoyancy - hydprms.weight
                    + (hydprms.linear_drag_z + hydprms.quadratic_drag_z * abs_smooth(v[3] - w_c; ϵ=1e-3)) * (v[3] - w_c)
                )

            f[8] = (1/(hydprms.inertia_z-hydprms.added_mass_yaw)) * (
                    ut[4]
                    + (v[2] - v_c)*(v[1] - u_c)*(hydprms.added_mass_y - hydprms.added_mass_x)
                    + (hydprms.linear_drag_yaw + hydprms.quadratic_drag_yaw * abs_smooth(v[4]; ϵ=1e-3)) * v[4]
                )

            f *= tdil
            return f
        end
    A_auto = (t, k, x, u, p, pbm) -> begin
        fx = xx -> f_impl(t, k, xx, u, p, pbm)
        ForwardDiff.jacobian(fx, x)
    end
    hydprms = pbm.mdl.hydroparams



    problem_set_dynamics!(
        pbm,
        # Dynamics f
        f_impl,
        # Jacobian df/dx
        A_auto,
        # Jacobian df/du
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            tdil = p[veh.id_t]
            B = zeros(pbm.nx, pbm.nu)
            B[5, 1] = (1/(hydprms.mass-hydprms.added_mass_x)) # dfu/dtau_x = (1/(mass-added_mass_x))
            B[6, 2] = (1/(hydprms.mass-hydprms.added_mass_y)) # dfv/dtau_y = (1/(mass-added_mass_y))
            B[7, 3] = (1/(hydprms.mass-hydprms.added_mass_z)) # dfw/dtau_z = (1/(mass-added_mass_z))
            B[8, 4] = (1/(hydprms.inertia_z-hydprms.added_mass_yaw)) # dfyaw/dtau_yaw = (1/(inertia_z - added_mass_yaw))
            B *= tdil
            return B
        end,
        # Jacobian df/dp
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            tdil = p[veh.id_t]
            F = zeros(pbm.nx, pbm.np)
            F[:, veh.id_t] = pbm.f(t, k, x, u, p)/tdil
            return F
        end)

    return nothing
end # function

function _common__set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (t, k, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj

        a = u[veh.id_u]
        tx = a[1]
        ty = a[2]
        tz = a[3]
        tyaw = a[4]

        tdil = p[veh.id_t]

        C = T_ConvexConeConstraint
        U = [C(veh.u_min-tx, :nonpos), # Input lower bound x u_min <= tx
             C(tx-veh.u_max, :nonpos), # Input upper bound x tx <= u_max
             C(veh.u_min-ty, :nonpos), # lower y
             C(ty-veh.u_max, :nonpos), # upper y
             C(veh.u_min-tz, :nonpos), # lower z
             C(tz-veh.u_max, :nonpos), # upper z
             C(veh.u_min-tyaw, :nonpos), # lower yaw
             C(tyaw-veh.u_max, :nonpos), #  upper yaw
             C(tdil-traj.tf_max, :nonpos), # Time dialation upper bound
             C(traj.tf_min-tdil, :nonpos)] # Time dialation lower bound 

        return U
        end)
    
    # Convex path constraints on the state
    problem_set_X!(
        pbm, (t, k, x, p, pbm) -> begin
        traj = pbm.mdl.traj
        veh = pbm.mdl.vehicle
        env = pbm.mdl.env
        u_robot = x[veh.id_v][1]
        v_robot = x[veh.id_v][2]
        z_pos = x[veh.id_r][3]
        C = T_ConvexConeConstraint
        X = [C(-u_robot, :nonpos), # Velocity u >= 0 positive
            # (v has to remain within a small band around 0, 
            # which depends on u since we want to move predominantly forward)
             C(-v_robot - 0.2*u_robot, :nonpos), # Velocity v >= -0.2*u 
             C(v_robot - 0.2*u_robot, :nonpos), # Velocity v <= 0.2*u 
            C(z_pos, :nonpos)]  # z always negative (below water)

        return X
        end)

    return nothing
end

function _common__set_nonconvex_constraints!(
    pbm::TrajectoryProblem,
    algo::T_Symbol)::Nothing

    # Constraint s
    _q__s = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        s = zeros(env.n_obs)
        for i = 1:env.n_obs
            # ---
            E = env.obs[i]
            r = x[veh.id_r]
            s[i] = 1-E(r)
            # ---
        end
        return s
    end

    # Jacobian ds/dx
    _q__C = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        C = zeros(env.n_obs, pbm.nx)
        for i = 1:env.n_obs
            # ---
            E = env.obs[i]
            r = x[veh.id_r]
            C[i, veh.id_r] = -∇(E, r)
            # ---
        end
        return C
    end

    if algo==:scvx
        problem_set_s!(pbm, algo, _q__s, _q__C)
    else
        _q___s = (t, k, x, p, pbm) -> _q__s(t, k, x, nothing, p, pbm)
        _q___C = (t, k, x, p, pbm) -> _q__C(t, k, x, nothing, p, pbm)
        problem_set_s!(pbm, algo, _q___s, _q___C)
    end

end

function _common__set_bcs!(pbm::TrajectoryProblem)::Nothing

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
