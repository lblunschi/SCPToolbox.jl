from pathlib import Path
import math
import subprocess
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def make_trajectory_video_box_realtime(
    base_dir=None,
    plot_name=None,
    xcol="x1",
    ycol="x2",
    yawcol="x4",
    tf=None,
    box_L=0.7,
    box_W=0.4,
    trail=True,
    trail_len=80,
    pad=0.2,
    show_start_goal=True,
    width_px=1920,
    height_px=1080,
    crf=18,
    preset="slow",
    fps_max=120,
    run = 0,
):

    base = Path(base_dir) if base_dir is not None else Path(__file__).parent
    data_dir = base / "data"
    run_dir = data_dir / f"case1"

    xd = pd.read_csv(run_dir / "data_xd.csv")
    p = pd.read_csv(run_dir / "data_p.csv")

    xs = xd[xcol].to_numpy()
    ys = xd[ycol].to_numpy()
    yaws = xd[yawcol].to_numpy()

    if "t" in xd.columns:
        tau = xd["t"].to_numpy()
    else:
        raise ValueError("solution_xd.csv must contain a 't' column")

    if tf is None:
        # assume first parameter value is final time, matching your Julia use
        tf = float(p["value"].iloc[0])

    nT = len(xs)
    dt = float(tf) / (nT - 1)
    fps = max(1, min(int(round(1.0 / dt)), fps_max))

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    dx = max(xmax - xmin, 2)
    dy = max(ymax - ymin, 2)
    xlims = (xmin - pad * dx, xmax + pad * dx)
    ylims = (ymin - pad * dy, ymax + pad * dy)

    frames_dir = Path(tempfile.mkdtemp())

    def box_vertices(x, y, psi):
        c = math.cos(psi)
        s = math.sin(psi)
        hl = box_L / 2.0
        hw = box_W / 2.0
        pts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
        out = []
        for bx, by in pts:
            wx = x + c * bx - s * by
            wy = y + s * bx + c * by
            out.append((wx, wy))
        return out

    figsize = (width_px / 100.0, height_px / 100.0)

    for k in range(nT):
        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # faint full path
        ax.plot(xs, ys, linewidth=2, alpha=0.15)

        # trail
        if trail:
            k0 = max(0, k - trail_len)
            ax.plot(xs[k0:k+1], ys[k0:k+1], linewidth=3)

        # start/goal
        if show_start_goal:
            ax.scatter([xs[0]], [ys[0]], s=80, marker="*", zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], s=60, marker="D", zorder=5)

        # vehicle box
        verts = box_vertices(xs[k], ys[k], yaws[k])
        poly = Polygon(verts, closed=True, alpha=0.6)
        ax.add_patch(poly)

        # heading line
        hx = xs[k] + (box_L / 2.0) * math.cos(yaws[k])
        hy = ys[k] + (box_L / 2.0) * math.sin(yaws[k])
        ax.plot([xs[k], hx], [ys[k], hy], linewidth=2)

        t_sec = float(tf) * tau[k]
        ax.set_title(f"t = {t_sec:.2f} s   (dt≈{dt:.3f} s, fps={fps:d})")

        outpng = frames_dir / f"frame_{k+1:06d}.png"
        fig.savefig(outpng)
        plt.close(fig)

    videos_dir = run_dir
    videos_dir.mkdir(parents=True, exist_ok=True)

    filename = "auv_traj_realtime.mp4" if plot_name is None else f"{plot_name}_traj_realtime.mp4"
    outfile = videos_dir / filename
    pattern = str(frames_dir / "frame_%06d.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(outfile),
    ]
    subprocess.run(cmd, check=True)

    return {
        "filename": str(outfile),
        "tf": float(tf),
        "dt": dt,
        "fps": fps,
        "frames_dir": str(frames_dir),
    }


if __name__ == "__main__":
    runs = [1]
    for run in runs:
        info = make_trajectory_video_box_realtime(
            plot_name="solution",
            xcol="x1",
            ycol="x2",
            yawcol="x4",
            run=run
        )
        print(info)