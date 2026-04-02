using CSV
using DataFrames

function export_solution_to_csv(sol; prefix="solution")

    # Base directory of THIS file
    base_dir = @__DIR__

    # Target folder: evaluation/<prefix>/
    out_dir = joinpath(base_dir, "evaluation/data", prefix)

    # Create directory if it doesn't exist
    mkpath(out_dir)

    N = size(sol.xd, 2)

    # ---- State ----
    xdf = DataFrame(t = vec(sol.td), k = 1:N)
    for i in 1:size(sol.xd, 1)
        xdf[!, "x$i"] = vec(sol.xd[i, :])
    end
    CSV.write(joinpath(out_dir, "data_xd.csv"), xdf)

    # ---- Control ----
    udf = DataFrame(t = vec(sol.td), k = 1:size(sol.ud, 2))
    for i in 1:size(sol.ud, 1)
        udf[!, "u$i"] = vec(sol.ud[i, :])
    end
    CSV.write(joinpath(out_dir, "data_ud.csv"), udf)

    # ---- Parameters ----
    pdf = DataFrame(
        param = ["p$i" for i in 1:length(sol.p)],
        value = vec(sol.p)
    )
    CSV.write(joinpath(out_dir, "data_p.csv"), pdf)

    println("Saved CSVs to: ", abspath(out_dir))

    # ---- Continuous Time ----
    if sol.xc !== missing && sol.uc !== missing

        tc = sol.xc.t   # time grid of continuous trajectory

        # sample states
        xc_vals = hcat([sample(sol.xc, t) for t in tc]...)

        # sample inputs
        uc_vals = hcat([sample(sol.uc, t) for t in tc]...)

        Nc = length(tc)

        xcdf = DataFrame(t = tc, k = 1:Nc)
        for i in 1:size(xc_vals, 1)
            xcdf[!, "x$i"] = vec(xc_vals[i, :])
        end
        CSV.write(joinpath(out_dir, "data_xc.csv"), xcdf)

        ucdf = DataFrame(t = tc, k = 1:Nc)
        for i in 1:size(uc_vals, 1)
            ucdf[!, "u$i"] = vec(uc_vals[i, :])
        end
        CSV.write(joinpath(out_dir, "data_uc.csv"), ucdf)
    end

    println("Saved CSVs to: ", abspath(out_dir))
end

function export_logs_to_txt(logs; prefix="solution")
    base_dir = @__DIR__
    out_dir = joinpath(base_dir, "evaluation", "data", prefix)
    mkpath(out_dir)
    file_path = joinpath(out_dir, "logs.txt")
    write(file_path, logs)
end