#= Tests for AUV flip.

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

using ECOS
using Printf
using Test

function run_optimization(csv_name="test")::Tuple{Float64, Bool}# x0, y0, z0, ψ0, xf, yf, zf, ψf, id_guess, kyaw, id, tf_guess, N, Nsub, iter_max, csv_name)::Tuple{Float64, Bool}
    mdl = AUVProblem()
    
    # yaw_final = ψf + 2*pi*kyaw
    
    # r0 = [x0, y0, z0, ψ0]
    # rf = [xf, yf, zf, yaw_final]
    
    # mdl.traj.r0 = r0
    # mdl.traj.rf = rf
    # mdl.traj.tf_guess = tf_guess
    # mdl.traj.use_guess = id_guess

    traj_pbm = TrajectoryProblem(mdl)
    define_problem!(traj_pbm, :scvx)

    # SCvx algorithm parameters
    N = 50
    Nsub = 100
    iter_max = 100
    disc_method = FOH
    λ = 1e2
    ρ_0 = 0.0
    ρ_1 = 0.2
    ρ_2 = 0.7
    β_sh = 1.5
    β_gr = 1.5
    η_init = 1.0
    η_lb = 1e-5
    η_ub = 1.0
    ε_abs = 1e-6
    ε_rel = 1e-4
    feas_tol = 1e-6
    q_tr = Inf
    q_exit = Inf
    solver = ECOS
    solver_options = Dict("verbose" => 0, "maxit" => 1000)
    pars = SCvx.Parameters(
        N,
        Nsub,
        iter_max,
        disc_method,
        λ,
        ρ_0,
        ρ_1,
        ρ_2,
        β_sh,
        β_gr,
        η_init,
        η_lb,
        η_ub,
        ε_abs,
        ε_rel,
        feas_tol,
        q_tr,
        q_exit,
        solver,
        solver_options,
    )
    
    # Create problem
    pbm = SCvx.create(pars, traj_pbm)

    # Solve problem
    sol, history = SCvx.solve(pbm)
    last_spbm = history.subproblems[end]
    export_solution_to_csv(sol; prefix=csv_name)
    scale = pbm.common.scale
    defect_max = -999
    Npoints = size(sol.xd,2)
    for ki = 1:Npoints-1
        max_defect_i = norm(scale.iSx * last_spbm.sol.defect[:, ki], Inf)
        if max_defect_i > defect_max
            defect_max = max_defect_i
        end
    end
    
    return defect_max, last_spbm.sol.feas
    
    # GC.gc()
    # return 999, false
end
