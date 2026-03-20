using JSON
using ECOS
using Printf
using Test
using LinearAlgebra
include("examples/auv_thruster/examples_auv_thruster_python.jl")
using .ExamplesAUVThrusterPython

const AP = ExamplesAUVThrusterPython.AUVThrusterPython

mutable struct SolverState
    mdl
    traj
    pars
end

function capture_logs_to_file(f, log_path::String)
    result = nothing

    open(log_path, "w") do io
        result = redirect_stdout(io) do
            redirect_stderr(io) do
                f()
            end
        end
        flush(io)
    end

    logs = read(log_path, String)
    return result, logs
end

function init_state()
    mdl = AP.AUVProblem()
    traj = AP.TrajectoryProblem(mdl)
    AP.define_problem!(traj, :scvx)

    N = 50
    Nsub = 100
    iter_max = 100
    disc_method = AP.FOH
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
    convex_solver = ECOS
    solver_options = Dict("verbose" => 0, "maxit" => 2000)

    pars = AP.SCvx.Parameters(
        N, Nsub, iter_max, disc_method, λ,
        ρ_0, ρ_1, ρ_2, β_sh, β_gr,
        η_init, η_lb, η_ub, ε_abs, ε_rel,
        feas_tol, q_tr, q_exit,
        convex_solver, solver_options,
    )

    return SolverState(mdl, traj, pars)
end

function run_case!(state::SolverState, x0::Float64, y0::Float64, z0::Float64, yaw0::Float64, xf::Float64, yf::Float64, zf::Float64, yawf::Float64; name::String="test")
    state.mdl.traj.r0 = [x0, y0, z0, yaw0]
    state.mdl.traj.rf = [xf, yf, zf, yawf]

    start_time = time()
    pbm = AP.SCvx.create(state.pars, state.traj)

    log_dir = joinpath(@__DIR__, "evaluation", "data", name)
    mkpath(log_dir)
    log_path = joinpath(log_dir, "logs.txt")

    (sol, history), logs = capture_logs_to_file(log_path) do
        AP.SCvx.solve(pbm)
    end

    last_spbm = history.subproblems[end]

    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            AP.export_solution_to_csv(sol; prefix=name)
        end
    end

    scale = pbm.common.scale
    defect_max = -Inf
    Npoints = size(sol.xd, 2)

    for ki in 1:Npoints-1
        max_defect_i = norm(scale.iSx * last_spbm.sol.defect[:, ki], Inf)
        defect_max = max(defect_max, max_defect_i)
    end

    solve_time = time() - start_time

    return Dict(
        "name" => name,
        "success" => last_spbm.sol.feas,
        "defect_max_scaled" => defect_max,
        "tf" => sol.p[1],
        "status" => string(sol.status),
        "comp_time" => solve_time,
    )
end

state = init_state()

while true
    line = readline(stdin)
    args = JSON.parse(line)

    x0, y0, z0, yaw0, xf, yf, zf, yawf, name_i, finished = args

    if finished
        break
    end

    result = run_case!(state, x0, y0, z0, yaw0, xf, yf, zf, yawf; name=name_i)
    println(JSON.json(result))
    flush(stdout)
end

println("terminated")
flush(stdout)