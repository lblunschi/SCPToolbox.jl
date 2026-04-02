function make_trajectory_video_box_realtime(sol;
    plot_name::Union{Nothing, String} = nothing,

    # indices in sol.xd
    xidx::Int=1,
    yidx::Int=2,
    yawidx::Int=4,

    # duration in seconds (your toolbox stores tf in sol.p[end])
    tf::Real,

    # box geometry
    box_L::Real=0.7,
    box_W::Real=0.4,

    # visuals
    trail::Bool=true,
    trail_len::Int=80,
    pad::Real=0.2,
    show_start_goal::Bool=true,

    # quality
    width_px::Int=1920,
    height_px::Int=1080,
    crf::Int=18,
    preset::String="slow",

    # fps cap (in case dt is tiny)
    fps_max::Int=120
)
    @assert isdefined(Main, :Plots) "Load Plots at top level: `using Plots`"
    @assert isdefined(Main, :Printf) "Load Printf at top level: `using Printf`"
    Plots.gr()  # pick backend explicitly

    X = sol.xd
    τ = sol.td              # normalized 0..1 grid in this toolbox
    nT = size(X, 2)

    xs = vec(X[xidx, :])
    ys = vec(X[yidx, :])
    yaws = vec(X[yawidx, :])

    # real-time dt and fps
    dt = float(tf) / (length(τ) - 1)
    fps = Int(clamp(round(1 / dt), 1, fps_max))

    # axis limits
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)
    dx = max(xmax - xmin, eps())
    dy = max(ymax - ymin, eps())
    xlims = (xmin - pad*dx, xmax + pad*dx)
    ylims = (ymin - pad*dy, ymax + pad*dy)

    frames_dir = mktempdir()

    # rotated box as a Shape
    function box_shape(x::Real, y::Real, ψ::Real)
        c, s = cos(ψ), sin(ψ)
        hl, hw = box_L/2, box_W/2
        pts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
        Xp = Float64[]; Yp = Float64[]
        for (bx, by) in pts
            wx = x + c*bx - s*by
            wy = y + s*bx + c*by
            push!(Xp, wx); push!(Yp, wy)
        end
        push!(Xp, Xp[1]); push!(Yp, Yp[1])
        return Plots.Shape(Xp, Yp)
    end

    for k in 1:nT
        plt = Plots.plot(;
            xlims=xlims, ylims=ylims,
            aspect_ratio=:equal,
            size=(width_px, height_px),
            legend=false,
            xlabel="x", ylabel="y"
        )

        # faint full path
        Plots.plot!(plt, xs, ys; lw=2, alpha=0.15)

        # trail
        if trail
            k0 = max(1, k - trail_len)
            Plots.plot!(plt, xs[k0:k], ys[k0:k]; lw=3)
        end

        # start/goal
        if show_start_goal
            Plots.scatter!(plt, [xs[1]], [ys[1]]; ms=7, marker=:star5)
            Plots.scatter!(plt, [xs[end]], [ys[end]]; ms=7, marker=:diamond)
        end

        # vehicle box + heading
        Plots.plot!(plt, box_shape(xs[k], ys[k], yaws[k]); fillalpha=0.6)
        hx = xs[k] + (box_L/2)*cos(yaws[k])
        hy = ys[k] + (box_L/2)*sin(yaws[k])
        Plots.plot!(plt, [xs[k], hx], [ys[k], hy]; lw=2)

        t_sec = float(tf) * τ[k]
        Plots.title!(plt, Printf.@sprintf("t = %.2f s   (dt≈%.3f s, fps=%d)", t_sec, dt, fps))

        outpng = joinpath(frames_dir, "frame_" * lpad(string(k), 6, '0') * ".png")
        Plots.savefig(plt, outpng)
    end

    figdir = joinpath(dirname(@__DIR__), "..", "videos")
    mkpath(figdir)   
    filename = plot_name === nothing ? "auv_traj_realtime.mp4" : "$(plot_name)_traj_realtime.mp4"
    outfile = joinpath(figdir, filename)
    pattern = joinpath(frames_dir, "frame_%06d.png")
    cmd = `ffmpeg -y -framerate $(fps) -i $(pattern) -c:v libx264 -preset $(preset) -crf $(crf) -pix_fmt yuv420p $(outfile)`
    run(cmd)

    return (filename=outfile, tf=float(tf), dt=dt, fps=fps, frames_dir=frames_dir)
end