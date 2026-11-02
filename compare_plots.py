"""
Plot and compare cart-pole runs saved by cartpole_sim.py.

Usage:
    python compare_plots.py results_LQR.npz results_PID.npz results_MPC.npz
    python compare_plots.py                      # picks up every results_*.npz

Outputs (light + dark variants, transparent background, for a site with dark mode):
    comparison_light.svg / .png
    comparison_dark.svg  / .png
    metrics.md
"""
import glob
import sys

import numpy as np
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

THEMES = {
    "light": dict(fg="#222222", grid="#000000", muted="#777777", push="#00000012"),
    "dark": dict(fg="#e6e6e6", grid="#ffffff", muted="#a0a0a0", push="#ffffff14"),
}

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------ metrics
def settling_time(t, err, tol, t_start=0.0):
    """Time after t_start at which |err| stays within tol for good."""
    m = t >= t_start
    t, err = t[m], err[m]
    outside = np.abs(err) > tol
    if not outside.any():
        return 0.0
    last = np.flatnonzero(outside)[-1]
    if last == len(t) - 1:
        return np.inf  # never settled
    return float(t[last + 1] - t_start)


def compute_metrics(run, tol_x=0.025, tol_theta=0.1):
    t, x, th, u, ref = run["t"], run["x"], run["theta"], run["u"], run["ref_x"]
    t_push = float(run["t_push"])
    return {
        "x settling time (s)": settling_time(t, x - ref, tol_x, t_push),
        "theta settling time (s)": settling_time(t, th, tol_theta, t_push),
        "peak theta deviation (deg)": float(np.degrees(np.max(np.abs(th)))),
        "peak force magnitude (N)": float(np.max(np.abs(u))),
        "control effort (integral u^2 dt)": float(_trapz(u ** 2, t)),
    }


def metrics_markdown(runs):
    names = list(runs)
    rows = {n: compute_metrics(r) for n, r in runs.items()}
    keys = list(next(iter(rows.values())))
    out = ["| Metric | " + " | ".join(names) + " |",
           "|---|" + "---|" * len(names)]
    for k in keys:
        out.append(f"| {k} | " + " | ".join(f"{rows[n][k]:.3f}" for n in names) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ plotting
def _style(theme):
    c = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": c["muted"],
        "axes.labelcolor": c["fg"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.color": c["fg"],
        "ytick.color": c["fg"],
        "text.color": c["fg"],
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })
    return c


def plot_comparison(runs, theme="light", prefix="comparison"):
    c = _style(theme)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True,
                             constrained_layout=True)
    fig.patch.set_alpha(0)

    for i, (name, r) in enumerate(runs.items()):
        col = COLORS[i % len(COLORS)]
        axes[0].plot(r["t"], np.degrees(r["theta"]), color=col, label=name)
        axes[1].plot(r["t"], r["x"], color=col)
        axes[2].plot(r["t"], r["u"], color=col)

    first = next(iter(runs.values()))
    axes[1].plot(first["t"], first["ref_x"], "--", color=c["muted"], lw=1.2,
                 label="reference")

    t_push, dur = float(first["t_push"]), float(first["dur"])
    for ax in axes:
        ax.set_facecolor("none")
        ax.grid(True, axis="y", color=c["grid"], alpha=0.15)
        ax.axvspan(t_push, t_push + max(dur, 0.05), color=c["push"], lw=0)
        ax.margins(x=0)
    axes[0].annotate("push", (t_push + dur / 2, 1.0),
                     xycoords=("data", "axes fraction"), ha="center", va="bottom",
                     fontsize=9, color=c["muted"])

    axes[0].set_ylabel("Pole angle (deg)")
    axes[1].set_ylabel("Cart position (m)")
    axes[2].set_ylabel("Control force (N)")
    axes[2].set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))

    for ext, kw in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{prefix}_{theme}.{ext}", transparent=True, **kw)
    plt.close(fig)


# ------------------------------------------------------------------ main
def load_runs(files):
    runs = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        name = str(d["name"])
        runs[name] = {k: d[k] for k in d.files}
    return runs


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("results_*.npz"))
    if not files:
        sys.exit("No results_*.npz files found.")
    runs = load_runs(files)
    for theme in THEMES:
        plot_comparison(runs, theme)
    md = metrics_markdown(runs)
    with open("metrics.md", "w") as fh:
        fh.write(md + "\n")
    print(md)