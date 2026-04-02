#= Tests for AUVSimplified flip.

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

function scvx_simplified()::Nothing

    # Problem definition
    mdl = AUVSimplifiedProblem()
    pbm = TrajectoryProblem(mdl)
    define_problem_simplified!(pbm, :scvx)

    # SCvx algorithm parameters
    N = 50
    Nsub = 50
    iter_max = 50
    disc_method = FOH
    λ = 10e3
    ρ_0 = 0.0
    ρ_1 = 0.1
    ρ_2 = 0.7
    β_sh = 2.0
    β_gr = 2.0
    η_init = 1.0
    η_lb = 1e-8
    η_ub = 10.0
    ε_abs = 1e-5
    ε_rel = 0.01 / 100
    feas_tol = 5e-3
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
    tf = [30, 40, 50, 60, 70]
    for i=1:5
        tf_i = tf[i]
        test_single_simplified(mdl, pbm, pars, SCvx,tf_i)
    end
    return nothing
end

function solve_simplified_for_guess(
    r0::AbstractVector,
    rf::AbstractVector,
    v0::AbstractVector,
    vf::AbstractVector;
    tf::Real = 50.0,
    N::Int = 50,
)
    # Problem definition
    mdl = AUVSimplifiedProblem()
    println("---- Simplified problem parameter override ----")
    println("r0: ", mdl.traj.r0, "  ->  ", r0)
    println("rf: ", mdl.traj.rf, "  ->  ", rf)
    println("v0: ", mdl.traj.v0, "  ->  ", v0)
    println("vf: ", mdl.traj.vf, "  ->  ", vf)
    println("tf: ", mdl.traj.tf, "  ->  ", tf)
    println("----------------------------------------------")
    mdl.traj.r0 .= r0
    mdl.traj.rf .= rf
    mdl.traj.v0 .= v0
    mdl.traj.vf .= vf
    mdl.traj.tf = tf

    traj = TrajectoryProblem(mdl)
    define_problem_simplified!(traj, :scvx)

    # SCvx algorithm parameters
    Nsub = 50
    iter_max = 50
    disc_method = FOH
    λ = 10e3
    ρ_0 = 0.0
    ρ_1 = 0.1
    ρ_2 = 0.7
    β_sh = 2.0
    β_gr = 2.0
    η_init = 1.0
    η_lb = 1e-8
    η_ub = 10.0
    ε_abs = 1e-5
    ε_rel = 0.01 / 100
    feas_tol = 5e-3
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

    spbm = SCvx.create(pars, traj)
    sol, history = SCvx.solve(spbm)
    # Solve problem
    last_spbm = history.subproblems[end]
    last_sol = last_spbm.sol

    @test sol.status == @sprintf("%s", SCP_SOLVED)


    energy_cost = sol.xd[9, end]

    @printf("Cost breakdown for last safe solution:\n")
    @printf("Energy cost = %.3e\n", energy_cost)

    @printf("")
    @printf("Original (L)      = %.3e\n", last_sol.L)
    @printf("Penalty  (L_pen)  = %.3e\n", last_sol.L_pen)
    @printf("Augmented (L_aug) = %.3e\n", last_sol.L_aug)
    @printf("Nonlinear (J_aug) = %.3e\n", last_sol.J_aug)

    return mdl, sol, history
end

"""
    test_single_simplified(pbm, traj, pars, solver)

Compute a single simplified trajectory.

# Arguments
- `mdl`: the AUVSimplified parameters.
- `traj`: the trajectory problem definition.
- `pars`: the algorithm parameters.
- `solver`: the solver algorithm's module.
"""
function test_single_simplified(
    mdl::AUVSimplifiedProblem,
    traj::TrajectoryProblem,
    pars::T,
    solver::Module,
    tf::Real,
)::Nothing where {T<:SCPParameters}

    mdl.traj.tf = tf
    test_heading(string(solver), "Single trajectory")

    # Create problem
    pbm = solver.create(pars, traj)

    # Solve problem
    sol, history = solver.solve(pbm)
    last_spbm = history.subproblems[end]
    last_sol = last_spbm.sol

    @test sol.status == @sprintf("%s", SCP_SOLVED)


    energy_cost = sol.xd[9, end]

    @printf("Cost breakdown for last safe solution:\n")
    @printf("Energy cost = %.3e\n", energy_cost)

    @printf("")
    @printf("Original (L)      = %.3e\n", last_sol.L)
    @printf("Penalty  (L_pen)  = %.3e\n", last_sol.L_pen)
    @printf("Augmented (L_aug) = %.3e\n", last_sol.L_aug)
    @printf("Nonlinear (J_aug) = %.3e\n", last_sol.J_aug)

    # Make plots
    try
        a = 1
        plot_convergence(history, "AUVSimplified")
        plot_trajectory_history(mdl, history)
        plot_final_trajectory_x_y(mdl, sol)
        plot_final_trajectory_x_z(mdl, sol)
        plot_final_trajectory_y_z(mdl, sol)     
        plot_inputs_ux_uy_uz_uyaw(mdl, sol)
        plot_velocity_xyzyaw_vs_time(mdl, sol)
        plot_position_xyzyaw_vs_time(mdl, sol)
        td_plot = sol.td
        xd_plot = sol.xd
        ud_plot = mdl.vehicle.thruster_allocation_matrix * sol.ud
        plot_dynamics_inputs_ux_uy_uz_uyaw_energy(mdl, td_plot, xd_plot, ud_plot)
    catch e
        showerror(stdout, e)
    end

    return nothing
end