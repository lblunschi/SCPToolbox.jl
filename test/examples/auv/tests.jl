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

function scvx()::Nothing

    # Problem definition
    mdl = AUVProblem()
    pbm = TrajectoryProblem(mdl)
    define_problem!(pbm, :scvx)

    # SCvx algorithm parameters
    N = 100
    Nsub = 100
    iter_max = 50
    disc_method = FOH
    λ = 5e2
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

    test_single(mdl, pbm, pars, SCvx)

    return nothing
end

function evaluate_energy_of_traj(ud::RealMatrix, p::RealVector)::Real

    energy = 0.0
    N = size(ud,2)
    tf = p[1]
    dt = tf / (N - 1)
    powers = [get_power_from_thrust(ud[:, i]) for i in 1:N]
    # u on segment k is linearly interpolated between knot k and k+1
    # e_seg_k = (1/2*p_k+1/2p_kp1)*dt
    energy = dt * (sum(powers) - 0.5*powers[1] - 0.5*powers[end])

    return energy
end


"""
    test_single(pbm, traj, pars, solver)

Compute a single trajectory.

# Arguments
- `mdl`: the auv parameters.
- `traj`: the trajectory problem definition.
- `pars`: the algorithm parameters.
- `solver`: the solver algorithm's module.
"""
function test_single(
    mdl::AUVProblem,
    traj::TrajectoryProblem,
    pars::T,
    solver::Module,
)::Nothing where {T<:SCPParameters}

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
        #plot_convergence(history, "AUV")
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
        e = evaluate_energy_of_traj(sol.ud, sol.p)
        println("Energy used in trajectory:", e)
    catch e
        showerror(stdout, e)
    end

    return nothing
end