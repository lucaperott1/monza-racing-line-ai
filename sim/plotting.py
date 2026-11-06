"""Shared drawing helpers: dark style, the track outline, corner labels and shading."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import paths
from .corners import CORNERS

BG = "#0f1115"
PANEL = "#171a21"
FG = "#e6e6e6"
MUTED = "#8a8f98"
TRACK = "#3a3f4b"
EDGE = "#d9d9d9"
BENCH = "#4fc3f7"   # benchmark colour
AI = "#ff7043"      # AI colour
FIRST = "#ab47bc"   # first finished lap


def dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#444a55", "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#2a2f3a",
        "axes.grid": True, "grid.linewidth": 0.6, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    })


def draw_track(ax, track, lw=0.8, fill=True):
    """Tarmac + white edge lines."""
    if fill:
        xs = np.concatenate([track.left_x, track.left_x[:1], track.right_x[::-1], track.right_x[-1:]])
        ys = np.concatenate([track.left_y, track.left_y[:1], track.right_y[::-1], track.right_y[-1:]])
        ax.fill(xs, ys, color=TRACK, lw=0, zorder=1)
    for ex, ey in ((track.left_x, track.left_y), (track.right_x, track.right_y)):
        ax.plot(np.append(ex, ex[0]), np.append(ey, ey[0]), color=EDGE, lw=lw, zorder=2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])


def corner_middle(track, a, b):
    i = track.index_at((a + b) / 2)
    return i


def label_corners(ax, track, push=170, size=10):
    """Corner names pushed outward, away from the track's middle."""
    cx, cy = track.X.mean(), track.Y.mean()
    for name, a, b in CORNERS:
        i = corner_middle(track, a, b)
        x, y = track.X[i], track.Y[i]
        dx, dy = x - cx, y - cy
        L = np.hypot(dx, dy)
        tx, ty = x + dx / L * push, y + dy / L * push
        ax.annotate(name, (x, y), (tx, ty), color=FG, fontsize=size, ha="center", va="center",
                    fontweight="bold", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))


def shade_corners(ax, label=True, km=False):
    """Grey bands where the corners are, on a distance axis."""
    f = 1000.0 if km else 1.0
    prev_mid, raised = -1e9, False
    for name, a, b in CORNERS:
        ax.axvspan(a / f, b / f, color="#ffffff", alpha=0.05, lw=0)
        mid = (a + b) / 2
        # corners close together (Lesmo 1 & 2): put the second name one row higher
        raised = (mid - prev_mid < 450) and not raised
        prev_mid = mid
        if label:
            ax.text(mid / f, 1.06 if raised else 1.01, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)


def zoom_to(ax, track, a, b, pad=40):
    i0, i1 = track.index_at(a), track.index_at(b)
    idx = np.arange(i0, i1 + 1) if i1 >= i0 else np.r_[i0:track.n, 0:i1 + 1]
    xs = np.r_[track.left_x[idx], track.right_x[idx]]
    ys = np.r_[track.left_y[idx], track.right_y[idx]]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def track_figure(track):
    """Picture 1: the track with its corners named."""
    fig, ax = plt.subplots(figsize=(9, 11))
    draw_track(ax, track)
    label_corners(ax, track)
    ax.plot([track.left_x[0], track.right_x[0]], [track.left_y[0], track.right_y[0]],
            color="#ff5252", lw=3, zorder=5)
    ax.annotate("Start / finish", (track.X[0], track.Y[0]), (track.X[0] + 230, track.Y[0] + 60),
                color="#ff5252", fontweight="bold", arrowprops=dict(arrowstyle="-", color="#ff5252"))
    j = 40
    ax.annotate("", (track.X[j + 12], track.Y[j + 12]), (track.X[j], track.Y[j]),
                arrowprops=dict(arrowstyle="-|>", color="#ff5252", lw=2, mutation_scale=20), zorder=5)
    ax.set_title(f"Monza - {track.length / 1000:.2f} km, driven clockwise")
    return save(fig, "1_monza_track.png")


def save(fig, name):
    path = os.path.join(paths.RESULTS, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(path, paths.ROOT))
    return path
