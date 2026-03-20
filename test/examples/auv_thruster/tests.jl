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

function multiple_settings()::Nothing

    # Problem definition
    #mdl = AUVProblem()
    tf_guess = [10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60]
    # update the boundary conditions for this run
    size_x = 20
    center_x = 0
    size_y = 20
    center_y = 0
    size_z = 10
    center_z = -5.5


    kyaw = [-1,0,1]
    for i=0:15
        x0 = rand() * size_x - size_x/2 + center_x
        y0 = rand() * size_y - size_y/2 + center_y
        z0 = rand() * size_z - size_z/2 + center_z
        ψ0 = rand() * 2*π - π
        xf = rand() * size_x - size_x/2 + center_x
        yf = rand() * size_y - size_y/2 + center_y
        zf = rand() * size_z - size_z/2 + center_z
        ψf = rand() * 2*π - π
        for j = 1:3
            for iguess = 1:2
                if iguess == 1
                    id_guess = 0
                else
                    id_guess = 4
                end
                id = 3*2*i+2*(j-1)+iguess
                
                test_id = "run_$(id)"
                println("Run: ", id)
                mdl = AUVProblem()
                
                ψf = -1.2802014188338235
                yaw_final = ψf + 2*pi*kyaw[j]
                
                r0 = [x0, y0, z0, ψ0]
                rf = [xf, yf, zf, yaw_final]
                rf = [9.517533251421687, 4.50596944352729, -6.32417431252921, yaw_final]
                r0 = [4.3398531983752635, -1.9266627369118465, -3.9677519098729483, -2.088607210759877]

                mdl.traj.r0 = r0
                mdl.traj.rf = rf
                mdl.traj.tf_guess = norm(rf[1:3]-r0[1:3])/0.3 #30 #tf_guess[id]
                mdl.traj.use_guess = id_guess

                pbm = TrajectoryProblem(mdl)
                define_problem!(pbm, :scvx)

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
                feas_tol = 1e-5
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
                try
                    e, tf = test_single(mdl, pbm, pars, SCvx; plot=true, plot_name=test_id)
                    
                catch err
                    println("Failed with error: ", err)
                end
                
                GC.gc()
            end
        end
    end    
    return nothing
end

function sweep_gamma_tf_max()::Nothing

    # Problem definition
    #mdl = AUVProblem()
    # gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # tf_max = [50, 60, 70, 80, 90, 100, 130, 150, 180, 200, 500]
    gammas = [0.4, 0.5, 0.6, 0.8]
    tf_max = [50, 70, 130, 200, 500]
    e_costs = zeros(length(gammas), length(tf_max))
    t_finish = zeros(length(gammas), length(tf_max))
    for i in eachindex(gammas)
        for j in eachindex(tf_max)
            # if j != 10
            #     continue
            # end
            # if i != 5
            #     continue
            # end
            println("Gamma i: ", i, "| Tf_max: ", j)
            mdl = AUVProblem()   # NEW every time
            gamma = gammas[i]
            tf_max_j = tf_max[j]
            # update gamma for this run
            mdl.traj.γ = gamma
            mdl.traj.tf_max = tf_max_j
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

            e, t_finish_i = test_single(mdl, pbm, pars, SCvx; plot=false)
            e_costs[i, j]  = e
            t_finish[i, j] = t_finish_i

            GC.gc()
        
        end
    end
    for i in eachindex(gammas)
        for j in eachindex(tf_max)
            println("Gamma: ", gammas[i], " tf_max: ", tf_max[j], " -> Energy: ", e_costs[i, j] , " W")
        end
    end
    
    return nothing
end

function evaluate_energy_of_traj(ud::RealMatrix, p::RealVector)::Real

    energy = 0.0
    N = size(ud,2)
    tf = p[1]
    dt = tf / (N - 1)
    powers = [get_power_from_thrust(ud[:, i]; real = true) for i in 1:N]
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
    solver::Module;
    plot=true,
    plot_name::Union{Nothing,String}=nothing,
)::Tuple{Real,Real} where {T<:SCPParameters}

    test_heading(string(solver), "Single trajectory")

    # Create problem
    pbm = solver.create(pars, traj)

    time_start = time()

    # Solve problem
    sol, history = solver.solve(pbm)
    scale = pbm.common.scale
    iSx = scale.iSx
    @show iSx
    if plot
        export_solution_to_csv(sol; prefix=plot_name)
    end
    duration = time() - time_start
    println("Took ", duration, " s to solve.")

    @test sol.status == @sprintf("%s", SCP_SOLVED)

    last_spbm = history.subproblems[end]
    last_sol = last_spbm.sol

    @printf("Cost breakdown for last safe solution:\n")

    @printf("")
    @printf("Original (L)      = %.3e\n", last_sol.L)
    @printf("Penalty  (L_pen)  = %.3e\n", last_sol.L_pen)
    @printf("Augmented (L_aug) = %.3e\n", last_sol.L_aug)
    @printf("Nonlinear (J_aug) = %.3e\n", last_sol.J_aug)

    #Make plots
    if plot
        try
            a = 1
            #plot_convergence(history, "AUV")
            plot_trajectory_history(mdl, history; plot_name)
            plot_final_trajectory_x_y(mdl, sol; plot_name)
            plot_final_trajectory_x_z(mdl, sol; plot_name)
            plot_final_trajectory_y_z(mdl, sol; plot_name)     
            plot_inputs_ux_uy_uz_uyaw(mdl, sol; plot_name)
            plot_velocity_xyzyaw_vs_time(mdl, sol; plot_name)
            plot_position_xyzyaw_vs_time(mdl, sol; plot_name)
            td_plot = sol.td
            xd_plot = sol.xd
            ud_plot = mdl.vehicle.thruster_allocation_matrix * sol.ud
            #plot_dynamics_inputs_ux_uy_uz_uyaw_energy(mdl, td_plot, xd_plot, ud_plot)
            # info = Base.invokelatest(
            #     make_trajectory_video_box_realtime, sol;
            #     plot_name=plot_name,
            #     xidx=1, yidx=2,
            #     yawidx=4,
            #     tf=sol.p[1],            
            #     width_px=1920, height_px=1080,
            #     crf=18, preset="slow",
            #     box_L=0.7, box_W=0.4
            # )
            # @show info
        catch e
            showerror(stdout, e)
        end
    end
    energy = evaluate_energy_of_traj(sol.ud, sol.p)
    println("Energy used in trajectory:", energy)
    history = nothing
    pbm = nothing
    t_finish = sol.p[1]
    return energy, t_finish
end