#= AUV obstacle avoidance example using SCvx.

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

using ECOS
using Printf
using LinearAlgebra
using Plots

include("common.jl")
include("../../models/auv.jl")
include("../../core/problem.jl")
include("../../core/scvx.jl")
include("create_vid.jl")
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = AUVProblem()
pbm = TrajectoryProblem(mdl)

define_problem!(pbm, :scvx)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: SCvx algorithm parameters ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 250 # Number of trajectory segments (N+1 knot points)
Nsub = 50 # Number of segments in each convex subproblem (Nsub+1 knot points)
iter_max = 200 # Maximum number of SCvx iterations
λ = 10000.0 # Penalty weight for virtual control and trust region violation
ρ_0 = 0.0 # Initial trust region radius
ρ_1 = 0.1 # Trust region radius expansion threshold
ρ_2 = 0.7 # Trust region radius contraction threshold
β_sh = 2.0 # Trust region radius expansion factor
β_gr = 2.0 # Trust region radius contraction factor
η_init = 0.3 # Initial virtual control and trust region violation tolerance
η_lb = 1e-3 # Minimum virtual control and trust region violation tolerance
η_ub = 10.0 # Maximum virtual control and trust region violation tolerance
ε_abs = 1e-5 # Absolute convergence tolerance
ε_rel = 0.01/100 # Relative convergence tolerance
feas_tol = 1e-3  # Feasibility tolerance for early termination (in case of infeasible problems)
q_tr = Inf # Maximum number of consecutive iterations with trust region violation above the threshold before termination
q_exit = Inf # Maximum number of consecutive iterations with virtual control and trust region violation below the threshold before termination
solver = ECOS 
solver_options = Dict("verbose"=>0)
pars = SCvxParameters(N, Nsub, iter_max, λ, ρ_0, ρ_1, ρ_2, β_sh, β_gr,
                      η_init, η_lb, η_ub, ε_abs, ε_rel, feas_tol, q_tr,
                      q_exit, solver, solver_options)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Solve trajectory generation problem ::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Number of trials. All trials will give the same solution, but we need many to
# plot statistically meaningful timing results
num_trials = 1

sol_list = Vector{SCPSolution}(undef, num_trials)
history_list = Vector{SCPHistory}(undef, num_trials)

for trial = 1:num_trials
    local scvx_pbm = SCvxProblem(pars, pbm)
    @printf("Trial %d/%d\n", trial, num_trials)
    if trial>1
        # Suppress output
        real_stdout = stdout
        (rd, wr) = redirect_stdout()
    end
    sol_list[trial], history_list[trial] = scvx_solve(scvx_pbm)
    if trial>1
        redirect_stdout(real_stdout)
    end
end

# Save one solution instance - for plotting a single trial
sol = sol_list[end]
history = history_list[end]

last_spbm = history.subproblems[end]
last_sol  = last_spbm.sol   # SCvxSubproblemSolution

vd  = last_sol.vd      # size: (nv, N-1)   (after E mapping in dynamics)
vs  = last_sol.vs      # size: (ns, N)
vic = last_sol.vic     # size: (nic,)
vtc = last_sol.vtc     # size: (ntc,)
P   = last_sol.P
Pf  = last_sol.Pf


function last_safe_subproblem(history::SCPHistory)
    n_rev = 0
    for spbm in reverse(history.subproblems)
        sol = spbm.sol
        if !sol.unsafe &&
           (sol.status == MOI.OPTIMAL || sol.status == MOI.LOCALLY_SOLVED) &&
           !any(isnan, sol.xd) &&
           !any(isinf, sol.xd)
            return sol, n_rev
        n_rev += 1
        end
    end
    error("No safe subproblem solution found in history.")
end

subsol_safe, n_reverse = last_safe_subproblem(history)
sol_safe = sol_list[end-n_reverse]


@printf("Original (L)      = %.3e\n", subsol_safe.L)
@printf("Penalty  (L_pen)  = %.3e\n", subsol_safe.L_pen)
@printf("Augmented (L_aug) = %.3e\n", subsol_safe.L_aug)
@printf("Nonlinear (J_aug) = %.3e\n", subsol_safe.J_aug)

@printf("max|E*vd| (stored as vd already mapped in cost via E): %.3e\n", norm(vd, Inf))
@printf("max vs:  %.3e\n", norm(vs, Inf))
@printf("max vic: %.3e\n", norm(vic, Inf))
@printf("max vtc: %.3e\n", norm(vtc, Inf))

@printf("‖vd‖₁ = %.3e, ‖vs‖₁ = %.3e, ‖vic‖₁ = %.3e, ‖vtc‖₁ = %.3e\n",
        norm(vd, 1), norm(vs, 1), norm(vic, 1), norm(vtc, 1))

X = sol.xd  # 8×50
@printf("xd size = %dx%d\n", size(X,1), size(X,2))

for t in 1:size(X,2)
    @printf("t=%2d  xd=%s\n", t, string(X[:,t]))
end
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Plot results :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
if num_trials > 1
    try
    plot_convergence(history_list, "AUV")

    catch err
        showerror(stderr, err, catch_backtrace())
    end
end


try
    plot_trajectory_history(mdl, history)
catch err
    showerror(stderr, err, catch_backtrace())
end
@info "Plotting final trajectory..."
plot_final_trajectory_x_y(mdl, sol_safe)
plot_final_trajectory_x_z(mdl, sol_safe)
plot_final_trajectory_y_z(mdl, sol_safe)
@info "Plotting final input..."
plot_inputs_ux_uy_uz_uyaw(mdl, sol_safe)

@info "Plotting convergence..."

plot_velocity_xyzyaw_vs_time(mdl, sol_safe)
plot_position_xyzyaw_vs_time(mdl, sol_safe)

# If your position is in states 1 and 2:
include("create_vid.jl")

info = Base.invokelatest(
    make_trajectory_video_box_realtime, sol_safe;
    filename="auv_traj_realtime.mp4",
    xidx=1, yidx=2,
    yawidx=4,
    tf=sol_safe.p[end],            # <-- real seconds
    width_px=1920, height_px=1080,
    crf=18, preset="slow",
    box_L=0.7, box_W=0.4
)
@show info