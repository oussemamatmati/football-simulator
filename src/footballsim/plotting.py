"""Matplotlib figures: pitch heatmaps, example possessions, calibration, hero image."""
from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import config as C
from .data import parse_token, zone_center


def _draw_pitch(ax, color="#cccccc"):
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, ec=color, lw=1.2))
    ax.plot([60, 60], [0, 80], color=color, lw=1)
    for x0 in (0, 102):
        ax.add_patch(Rectangle((x0, 18), 18, 44, fill=False, ec=color, lw=1))
    ax.set_xlim(-2, 122)
    ax.set_ylim(82, -2)  # y axis down, like StatsBomb
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def zone_heatmap(ax, dist: np.ndarray, title: str, vmax: float | None = None, cmap: str = "viridis"):
    im = ax.imshow(
        dist.reshape(C.GRID_X, C.GRID_Y).T, extent=[0, 120, 80, 0], cmap=cmap,
        vmin=0, vmax=vmax, aspect="equal", alpha=0.9,
    )
    _draw_pitch(ax, color="white")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("attacking direction →", fontsize=8)
    return im


def plot_possession(ax, toks: list[str], title: str | None = None):
    """Draw one possession as arrows between zone centres (red stars = shots)."""
    _draw_pitch(ax)
    pts = []
    for t in toks:
        parsed = parse_token(t)
        if parsed:
            pts.append((zone_center(parsed[2]), parsed[0]))
    for k in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[k][0], pts[k + 1][0]
        jitter = 0.8 * (k % 3 - 1)
        ax.annotate("", xy=(x1, y1 + jitter), xytext=(x0, y0 + jitter),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2, alpha=0.8))
    for (x, y), action in pts:
        shot = action == "Shot"
        ax.scatter([x], [y], s=70 if shot else 22, c="#d62728" if shot else "#1f77b4",
                   marker="*" if shot else "o", zorder=3)
    if title:
        ax.set_title(title, fontsize=9)


def calibration_plot(ax, calib: dict):
    p, y = np.array(calib["p"]), np.array(calib["y"])
    hi = float(max(p.max(), y.max()))
    ax.plot([0, hi], [0, hi], "--", color="gray", label="perfect calibration")
    ax.plot(p, y, "o-", label="model")
    ax.set_xlabel("predicted P(next event = shot)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability")
    ax.legend()


def hero_figure(report: dict, real_dist: np.ndarray, sim_dist: np.ndarray):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    vmax = float(max(real_dist.max(), sim_dist.max()))
    im = zone_heatmap(ax[0], real_dist, "Real possessions (held-out matches)", vmax)
    zone_heatmap(ax[1], sim_dist, "Simulated by transformer", vmax)
    fig.colorbar(im, ax=ax[:2], shrink=0.8, label="share of events")
    calibration_plot(ax[2], report["calibration"])
    fig.suptitle(
        f"Football possession simulator | perplexity {report['transformer_perplexity']:.1f} vs Markov "
        f"{report['markov_perplexity']:.1f} ({report['improvement_pct']:.0f}% lower)", y=1.02,
    )
    return fig
