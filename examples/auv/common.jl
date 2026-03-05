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

    problem_set_dims!(pbm, 8, 5, 1) # nx, nu, np

    return nothing
end

function _common__set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl

    tdil_min = mdl.traj.tf_min
    tdil_max = mdl.traj.tf_max
    tdil_max_adj = tdil_min+1.0*(tdil_max-tdil_min)
    problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t,
                          (tdil_min, tdil_max_adj))

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
        return γ*(tdil/tdil_max)^2
        end)

    # Running cost
    if algo==:scvx
        problem_set_running_cost!(
            pbm, algo,
            (x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj
            γ = traj.γ
            return sum(u[i]^2 for i in 1:4)
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
    return sqrt(x^2 + ϵ^2)
end

"""
    get_current_body_smooth_and_partials(state; yawidx=4, shear=0.2, width=0.5)

Returns (u_c, v_c, w_c, du_dx, dv_dx, du_dψ, dv_dψ)

All derivatives are w.r.t. inertial x and yaw ψ (treating other states constant).
"""
function get_current_body_smooth_and_partials(state::AbstractVector;
    yawidx::Int=4,
    shear::Real=0.2,
    width::Real=0.5
)
    x = state[1]
    ψ = state[yawidx]

    # inertial shear in +/−y (smooth)
    v_i = -shear * tanh(x / width)

    sψ = sin(ψ)
    cψ = cos(ψ)

    # inertial -> body (u_i = 0)
    u_c = sψ * v_i
    v_c = cψ * v_i
    w_c = 0.0

    # dv_i/dx
    a = x / width
    sech2 = 1 / (cosh(a)^2)
    dv_i_dx = -(shear / width) * sech2

    # partials
    du_dx = sψ * dv_i_dx
    dv_dx = cψ * dv_i_dx

    du_dψ =  cψ * v_i
    dv_dψ = -sψ * v_i

    return u_c, v_c, w_c, du_dx, dv_dx, du_dψ, dv_dψ
end

function _common__set_dynamics!(pbm::TrajectoryProblem)::Nothing
    # Current velocity (assumed zero for now, TODO implement current models)
    # Irrotational current, so no yaw component
    u_c = 0.0
    v_c = 0.0
    w_c = 0.0
    
    hydprms = pbm.mdl.hydroparams



    problem_set_dynamics!(
        pbm,
        # Dynamics f
        (t, k, x, u, p, pbm) -> begin
            u_c, v_c, w_c, du_dx, dv_dx, du_dψ, dv_dψ =
                get_current_body_smooth_and_partials(x)
            g = pbm.mdl.env.g
            veh = pbm.mdl.vehicle
            pos = x[veh.id_r] # Position x,y,z, yaw (psi)
            v = x[veh.id_v] # Velocity u,v,w, yaw rate (r)
            ut = u[veh.id_u] # Control input (thrust) x,y,z,yaw
            tdil = p[veh.id_t]
            f = zeros(pbm.nx)
            
            f[1] = cos(pos[4])*v[1]-sin(pos[4])*v[2] # xdot = cos(yaw)*u -sin(yaw)*v
            f[2] = sin(pos[4])*v[1]+cos(pos[4])*v[2] # ydot = sin(yaw)*u + cos(yaw)*v
            f[3] = v[3] # zdot = w
            f[4] = v[4] # yawdot = r
            f[5] = (1/(hydprms.mass-hydprms.added_mass_x)) * (ut[1] + (v[2] - v_c)*(hydprms.mass*v[4] - hydprms.added_mass_y*v[4]) + (hydprms.linear_drag_x + hydprms.quadratic_drag_x * abs_smooth((v[1] - u_c); ϵ=1e-3)) * (v[1] - u_c)) # udot = (1/(mass-added_mass_x))*(tau_x + (v - vc)(m*r - added_mass_y*r) + (linear_drag_x + quadratic_drag_x*|(u-uc)|)*(u-uc))
            f[6] = (1/(hydprms.mass-hydprms.added_mass_y)) * (ut[2] + (v[1] - u_c)*(hydprms.added_mass_x*v[4] - hydprms.mass*v[4]) + (hydprms.linear_drag_y + hydprms.quadratic_drag_y * abs_smooth((v[2] - v_c); ϵ=1e-3)) * (v[2] - v_c)) # vdot = (1/(mass-added_mass_y))*(tau_y + (u - uc)(added_mass_x*r - m*r) + (linear_drag_y + quadratic_drag_y*|(v-vc)|)*(v-vc))
            f[7] = (1/(hydprms.mass-hydprms.added_mass_z)) * (ut[3] + hydprms.buoyancy - hydprms.weight + (hydprms.linear_drag_z + hydprms.quadratic_drag_z * abs_smooth((v[3] - w_c); ϵ=1e-3)) * (v[3] - w_c)) # wdot = (1/(mass-added_mass_z))*(tau_z - weight + buoyancy + (linear_drag_z + quadratic_drag_z*|w-wc|)*(w-wc))
            f[8] = (1/(hydprms.inertia_z-hydprms.added_mass_yaw)) * (ut[4] + (v[2] - v_c)*(v[1] - u_c)*(hydprms.added_mass_y - hydprms.added_mass_x) + (hydprms.linear_drag_yaw + hydprms.quadratic_drag_yaw * abs_smooth(v[4]; ϵ=1e-3)) * v[4]) # rdot = (1/(inertia_z - added_mass_yaw))*(tau_yaw + (v-vc)(u-uc)(added_mass_y-added_mass_x) + (linear_drag_yaw + quadratic_drag_yaw*|r|)*r)
        
            f *= tdil
            return f
        end,
        # Jacobian df/dx
        (t, k, x, u, p, pbm) -> begin

            u_c, v_c, w_c, du_dx, dv_dx, du_dψ, dv_dψ =
                get_current_body_smooth_and_partials(x)
            veh = pbm.mdl.vehicle
            tdil = p[veh.id_t]
            pos = x[veh.id_r] # Position x,y,z, yaw (psi)
            v = x[veh.id_v] # Velocity u,v,w, yaw rate (r)
            A = zeros(pbm.nx, pbm.nx)
            # dfx
            A[1, 4] = -sin(pos[4])*v[1]-cos(pos[4])*v[2] # dfx/dyaw = -sin(yaw)*u - cos(yaw)*v
            A[1, 5] = cos(pos[4]) # dfx/du = cos(yaw)
            A[1, 6] = -sin(pos[4]) # dfx/dv = -sin(yaw)
            # dfy
            A[2, 4] = cos(pos[4])*v[1]-sin(pos[4])*v[2] # dfy/dyaw = cos(yaw)*u - sin(yaw)*v
            A[2, 5] = sin(pos[4]) # dfy/du = sin(yaw)
            A[2, 6] = cos(pos[4]) # dfy/dv = cos(yaw
            # dfz
            A[3, 7] = 1.0 # dfz/dw = 1
            # dfyaw
            A[4, 8] = 1.0 # dfyaw/dr = 1
            # dfu
            # TODO implement jacobian for current model that depends on position
            denx = hydprms.mass - hydprms.added_mass_x
            Kx   = (hydprms.mass - hydprms.added_mass_y) * v[4]   # (m - A_y) r

            urx = v[1] - u_c
            a   = sqrt(urx^2 + 1e-3^2)
            dD  = hydprms.linear_drag_x + hydprms.quadratic_drag_x * (a + urx^2 / a)

            # ∂u̇/∂x  from current
            A[5,1] = (1/denx) * ( (-dv_dx) * Kx  + dD * (-(du_dx)) )

            # ∂u̇/∂ψ  from current
            A[5,4] = (1/denx) * ( (-dv_dψ) * Kx  + dD * (-(du_dψ)) )
            """A[5, 1] = 0.0 # dfu/dx = 0
            A[5, 2] = 0.0 # dfu/dy = 0
            A[5, 3] = 0.0 # dfu/dz = 0 
            A[5, 4] = 0.0 # dfu/dyaw = 0"""
            denx = hydprms.mass - hydprms.added_mass_x
            ur = v[1] - u_c
            a  = sqrt(ur^2 + 1e-3^2)
            dD = hydprms.linear_drag_x + hydprms.quadratic_drag_x * (a + ur^2 / a)

            A[5,5] = (1/denx) * dD
            #A[5, 5] = (1/(hydprms.mass-hydprms.added_mass_x)) * (hydprms.linear_drag_x + 2*hydprms.quadratic_drag_x * abs_smooth((v[1] - u_c); ϵ=1e-3)) # dfu/du = (1/(mass-added_mass_x))*(linear_drag_x + 2*quadratic_drag_x*|(u-uc)|)
            A[5, 6] = (1/(hydprms.mass-hydprms.added_mass_x)) * (hydprms.mass*v[4] - hydprms.added_mass_y*v[4]) # dfu/dv = (1/(mass-added_mass_x))*(m*r - added_mass_y*r)
            A[5, 8] = (1/(hydprms.mass-hydprms.added_mass_x)) * (v[2] - v_c)*(hydprms.mass - hydprms.added_mass_y) # dfu/dr = (1/(mass-added_mass_x))*(v-vc)(m - added_mass_y)
            # dfv
            # TODO implement jacobian for current model that depends on position
            deny = hydprms.mass - hydprms.added_mass_y
            Ky   = (hydprms.added_mass_x - hydprms.mass) * v[4]   # (A_x - m) r

            ury = v[2] - v_c
            a   = sqrt(ury^2 + 1e-3^2)
            dD  = hydprms.linear_drag_y + hydprms.quadratic_drag_y * (a + ury^2 / a)

            A[6,1] = (1/deny) * ( (-(du_dx)) * Ky  + dD * (-(dv_dx)) )
            A[6,4] = (1/deny) * ( (-(du_dψ)) * Ky  + dD * (-(dv_dψ)) )
            """A[6, 1] = 0.0 # dfv/dx = 0
            A[6, 2] = 0.0 # dfv/dy = 0
            A[6, 3] = 0.0 # dfv/d
            A[6, 4] = 0.0 # dfv/dyaw = 0"""
            A[6, 5] = (1/(hydprms.mass-hydprms.added_mass_y)) * (hydprms.added_mass_x*v[4] - hydprms.mass*v[4]) # dfv/du = -(1/(mass-added_mass_y))*(added_mass_x*r - m*r)
            deny = hydprms.mass - hydprms.added_mass_y
            ur = v[2] - v_c
            a  = sqrt(ur^2 + 1e-3^2)
            dD = hydprms.linear_drag_y + hydprms.quadratic_drag_y * (a + ur^2 / a)

            A[6,6] = (1/deny) * dD
            #A[6, 6] = (1/(hydprms.mass-hydprms.added_mass_y)) * (hydprms.linear_drag_y + 2*hydprms.quadratic_drag_y*abs_smooth((v[2]-v_c); ϵ=1e-3)) # dfv/dv = (1/(mass-added_mass_y))*(linear_drag_y + 2*quadratic_drag_y*|(v-vc)|)
            A[6, 8] = (1/(hydprms.mass-hydprms.added_mass_y)) * (v[1] - u_c)*(hydprms.added_mass_x - hydprms.mass) # dfv/dr = (1/(mass-added_mass_y))*(v-vc)(added_mass_x - m)
            # dfw
            # TODO implement jacobian for current model that depends on position
            A[7, 1] = 0.0 # dfw/dx = 0
            A[7, 2] = 0.0 # dfw/dy = 0
            A[7, 3] = 0.0 # dfw/dz = 0
            A[7, 4] = 0.0 # dfw/dyaw = 0
            denz = hydprms.mass - hydprms.added_mass_z
            ur = v[3] - w_c
            a  = sqrt(ur^2 + 1e-3^2)
            dD = hydprms.linear_drag_z + hydprms.quadratic_drag_z * (a + ur^2 / a)

            A[7,7] = (1/denz) * dD
            #A[7, 7] = (1/(hydprms.mass-hydprms.added_mass_z)) * (hydprms.linear_drag_z + 2*hydprms.quadratic_drag_z * abs_smooth((v[3] - w_c); ϵ=1e-3)) # dfw/dw = (1/(mass-added_mass_z))*(linear_drag_z + 2*quadratic_drag_z*|w-wc|)
            # dfyaw
            # TODO implement jacobian for current model that depends on position
            denr = hydprms.inertia_z - hydprms.added_mass_yaw
            Δam  = (hydprms.added_mass_y - hydprms.added_mass_x)

            term_u = (v[1] - u_c)
            term_v = (v[2] - v_c)

            A[8,1] = (1/denr) * Δam * ( (-dv_dx) * term_u + term_v * (-(du_dx)) )
            A[8,4] = (1/denr) * Δam * ( (-dv_dψ) * term_u + term_v * (-(du_dψ)) )
            """A[8, 1] = 0.0 # dfyaw/dx = 0
            A[8, 2] = 0.0 # dfyaw/dy = 0
            A[8, 3] = 0.0 # dfyaw/dz = 0
            A[8, 4] = 0.0 # dfyaw/dyaw = 0"""
            A[8, 5] = (1/(hydprms.inertia_z-hydprms.added_mass_yaw)) * (v[2] - v_c)*(hydprms.added_mass_y - hydprms.added_mass_x) # dfyaw/du = (1/(inertia_z - added_mass_yaw))*(v-vc)(added_mass_y - added_mass_x)
            A[8, 6] = (1/(hydprms.inertia_z-hydprms.added_mass_yaw)) * (v[1] - u_c)*(hydprms.added_mass_y - hydprms.added_mass_x) # dfyaw/dv = (1/(inertia_z - added_mass_yaw))*(v-vc)(added_mass_y - added_mass_x)
            denr = hydprms.inertia_z - hydprms.added_mass_yaw
            ur = v[4]
            a  = sqrt(ur^2 + 1e-3^2)
            dD = hydprms.linear_drag_yaw + hydprms.quadratic_drag_yaw * (a + ur^2 / a)

            A[8,8] = (1/denr) * dD
            #A[8, 8] = (1/(hydprms.inertia_z-hydprms.added_mass_yaw)) * (hydprms.linear_drag_yaw + 2*hydprms.quadratic_drag_yaw * abs_smooth(v[4]; ϵ=1e-3)) # dfyaw/dr = (1/(inertia_z - added_mass_yaw))*(linear_drag_yaw + 2*quadratic_drag_yaw*|r|)
            A *= tdil
            return A
        end,
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
