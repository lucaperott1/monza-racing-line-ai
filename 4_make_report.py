"""4 - Turn the training history into pictures and numbers in results/.

1_monza_track.png        the track and its corners
2_racing_lines.png       test laps on the track, coloured by how much practice the AI had
3_learning_curve.png     how far each test lap got, and the lap times
4_telemetry.png          speed, pedals and time gained/lost against the benchmark
5_grip_usage.png         how much of the tyres' grip the AI uses
6_line_vs_benchmark.png  the AI's best line next to the benchmark line
summary.txt              the key numbers
(Optional: --name seed2 for an AI trained under another name.)
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from sim import ai, paths
from sim import car as C
from sim import plotting as P
from sim.corners import CORNERS
from sim.optimal import fmt_time, get_optimal
from sim.track import get_track

KMH = 3.6
ZOOMS6 = ["Rettifilo", "Roggia", "Lesmo 1", "Lesmo 2", "Ascari", "Parabolica"]
CORNER = {n: (a, b) for n, a, b in CORNERS}


def arr(lap, k):
    return np.asarray(lap["tel"][k], float)


def mono(d):
    return np.maximum.accumulate(d)


def best_lap(laps):
    best = None
    for lap in laps:
        if ai.is_better(lap, best):
            best = lap
    return best


def lap_label(lap):
    if lap["finished"]:
        return f"{fmt_time(lap['lap_time'])} at {lap['steps'] / 1e6:.2f}M steps"
    return f"furthest attempt ({100 * lap['frac']:.0f}%) at {lap['steps'] / 1e6:.2f}M steps"


def speed_line(ax, x, y, v, lw=2.2, norm=None, zorder=4):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="turbo", norm=norm or Normalize(50, 350), lw=lw, zorder=zorder)
    lc.set_array(v[:-1] * KMH)
    ax.add_collection(lc)
    return lc


# ----------------------------------------------------------------- 2 racing lines
def fig_racing_lines(track, laps, bench):
    pick = laps if len(laps) <= 60 else [laps[int(i)] for i in np.linspace(0, len(laps) - 1, 60)]
    smax = max(l["steps"] for l in laps) / 1e6
    norm = Normalize(0, smax)
    cmap = plt.get_cmap("plasma")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    for k, ax in enumerate(axes):
        P.draw_track(ax, track, lw=0.8 if k == 0 else 1.2)
        if k > 0:
            ax.plot(bench["line_x"] + bench["line_x"][:1], bench["line_y"] + bench["line_y"][:1],
                    "--", color=P.BENCH, lw=1.5, zorder=5, label="benchmark line")
        for lap in pick:
            col = cmap(norm(lap["steps"] / 1e6))
            ax.plot(arr(lap, "x"), arr(lap, "y"), color=col, lw=0.9 if k == 0 else 1.3, alpha=0.8, zorder=3)
            if not lap["finished"]:
                ax.plot(lap["end_x"], lap["end_y"], "x", color=col, ms=8 if k else 6, mew=2, zorder=6)
    P.label_corners(axes[0], track, size=9)
    which = ("all test laps" if len(pick) == len(laps) else
             f"{len(pick)} of {len(laps)} test laps, evenly spaced from first to last")
    axes[0].set_title(f"{which}\ncoloured by practice  (x = went off / stopped)", pad=26)
    for ax, name in ((axes[1], "Rettifilo"), (axes[2], "Ascari")):
        P.zoom_to(ax, track, *CORNER[name], pad=25)
        ax.set_title(f"Zoom: {name}")
        ax.legend(loc="lower right", fontsize=8)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("practice steps (millions)")
    fig.suptitle("How the AI's racing line developed", fontsize=15, fontweight="bold")
    return P.save(fig, "2_racing_lines.png")


# ----------------------------------------------------------------- 3 learning curve
def fig_learning(track, laps, bench):
    """Picture 3 (original design): how far each test lap got, and the lap times."""
    steps = np.array([l["steps"] for l in laps]) / 1e6
    frac = np.array([l["frac"] for l in laps]) * 100
    fin = np.array([l["finished"] for l in laps])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    a1.plot(steps, frac, color="#555b66", lw=0.8, zorder=1)
    a1.scatter(steps[~fin], frac[~fin], s=14, color="#ef5350", label="went off / stopped", zorder=3)
    a1.scatter(steps[fin], frac[fin], s=14, color="#66bb6a", label="finished the lap", zorder=3)
    for name, a, b in CORNERS:
        y = 100 * a / track.length
        a1.axhline(y, color=P.MUTED, lw=0.5, ls=":")
        a1.text(steps.max() * 1.005 if len(steps) else 0, y, name, color=P.MUTED, fontsize=8, va="center")
    a1.set_ylim(0, 105)
    a1.set_ylabel("% of the lap completed")
    a1.set_title("How far each test lap got")
    a1.legend(loc="lower right")

    if fin.any():
        t = np.array([l["lap_time"] if l["finished"] else np.nan for l in laps])
        a2.scatter(steps[fin], t[fin], s=16, color=P.AI, label="test lap time", zorder=3)
        bsf = np.fmin.accumulate(np.where(fin, t, np.inf))
        ok = np.isfinite(bsf)
        a2.step(steps[ok], bsf[ok], where="post", color="#ffd54f", lw=2, label="best so far")
        a2.axhline(bench["standing_lap"], color=P.BENCH, ls="--", lw=1.5,
                   label=f"benchmark {fmt_time(bench['standing_lap'])}")
        lo = min(bench["standing_lap"], np.nanmin(t)) - 2
        hi = min(np.nanmax(t), np.nanmin(t) + 40) + 2
        a2.set_ylim(lo, hi)
        a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_time(v)[:-2]))
        a2.legend(loc="upper right")
    else:
        a2.text(0.5, 0.5, "No test lap has finished yet - keep training!", transform=a2.transAxes,
                ha="center", va="center", fontsize=14, color=P.MUTED)
    a2.set_ylabel("lap time (standing start)")
    a2.set_xlabel("practice steps (millions)")
    a2.set_title("Lap times of finished test laps")
    fig.suptitle("Learning curve", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return P.save(fig, "3_learning_curve.png")


ALT = "alternative_versions"   # redesigned pictures 3 and 5 are saved in results/alternative_versions/


def fig_learning_alt(track, laps, bench):
    """Picture 3, alternative design: outcome bars + lap-time milestones."""
    steps = np.array([l["steps"] for l in laps]) / 1e6
    fin = np.array([l["finished"] for l in laps])
    first_corner = CORNERS[0][0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.7]})

    # --- top: what happened in the test laps, period by period (stacked bars)
    smax = max(steps.max(), 0.05)
    bw = next(w for w in (0.1, 0.25, 0.5, 1, 2, 5, 10, 20) if smax / w <= 40)
    edges = np.arange(0, smax + bw, bw)
    at_first = np.array([(not l["finished"]) and first_corner in l["where"] for l in laps])
    rows = {"finished the lap": [], f"went off at the {first_corner} (first chicane)": [],
            "went off / stopped elsewhere": []}
    centres = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (steps > lo) & (steps <= hi)
        if not m.any():
            continue
        n = m.sum()
        centres.append((lo + hi) / 2)
        rows["finished the lap"].append(100 * fin[m].sum() / n)
        rows[f"went off at the {first_corner} (first chicane)"].append(100 * at_first[m].sum() / n)
        rows["went off / stopped elsewhere"].append(100 * (n - fin[m].sum() - at_first[m].sum()) / n)
    bottom = np.zeros(len(centres))
    for (label, vals), col in zip(rows.items(), ("#66bb6a", "#ef5350", "#ffa726")):
        a1.bar(centres, vals, width=bw * 0.92, bottom=bottom, color=col, label=label, lw=0)
        bottom += np.array(vals)
    a1.set_ylim(0, 100)
    a1.set_ylabel("share of test laps (%)")
    a1.set_title(f"What happened in the test laps (each bar = {bw:g} million steps)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9.5)
    a1.grid(axis="x", visible=False)

    # --- bottom: lap times with milestones
    bench_t = bench["standing_lap"]
    if fin.any():
        t = np.array([l["lap_time"] if l["finished"] else np.nan for l in laps])
        bsf = np.fmin.accumulate(np.where(fin, t, np.inf))
        ok = np.isfinite(bsf)
        top = min(np.nanmax(t), bench_t + 45)
        x_end = steps[-1]
        sx = np.r_[steps[ok], x_end]
        sy = np.r_[bsf[ok], bsf[ok][-1]]
        a2.fill_between(sx, bench_t, sy, step="post", color="#ffd54f", alpha=0.10, lw=0)
        a2.scatter(steps[fin], t[fin], s=10, color=P.AI, alpha=0.35, lw=0, label="each finished test lap")
        a2.step(sx, sy, where="post", color="#ffd54f", lw=3, label="best lap so far")
        a2.axhline(bench_t, color=P.BENCH, ls="--", lw=1.8)
        a2.text(steps[0], bench_t - 0.8, f"benchmark (perfect driver) {fmt_time(bench_t)}", color=P.BENCH,
                ha="left", va="top", fontsize=10, fontweight="bold")
        # milestones: first lap, first time under each round number, best
        i_first = int(np.argmax(fin))
        marks = [(i_first, f"first finished lap {fmt_time(t[i_first])}")]
        for limit in (120, 110, 100, 95, 90):
            under = np.where(ok & (bsf < limit))[0]
            if len(under) and t[i_first] >= limit and under[0] != i_first:
                marks.append((under[0], f"under {fmt_time(limit)[:-4]}"))
        # milestones reached almost together: keep only the later one
        marks = [m for k, m in enumerate(marks)
                 if k + 1 == len(marks) or steps[marks[k + 1][0]] - steps[m[0]] > 0.6 or k == 0]
        i_best = int(np.nanargmin(t))
        for i, text in marks:
            y = min(bsf[i], top - 1.5)
            a2.plot(steps[i], y, "o", color="white", ms=6, zorder=6)
            at_top = y >= top - 2
            a2.annotate(f"{text}  @ {steps[i]:.1f}M steps", (steps[i], y), (10, -8) if at_top else (6, 7),
                        textcoords="offset points", color="white", fontsize=9,
                        va="top" if at_top else "bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc=P.PANEL, ec="none", alpha=0.85))
        a2.plot(steps[i_best], t[i_best], "*", color="#ffd54f", ms=18, mec="black", zorder=7)
        a2.annotate(f"BEST {fmt_time(t[i_best])} @ {steps[i_best]:.2f}M steps\n"
                    f"{t[i_best] - bench_t:+.2f} s to the benchmark",
                    (steps[i_best], t[i_best]), (30, 55), textcoords="offset points", ha="left",
                    color="#ffd54f", fontsize=10.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#ffd54f"))
        a2.set_ylim(bench_t - 6, top)
        a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_time(v)[:-4]))
        a2.legend(loc="upper right", fontsize=9.5)
    else:
        a2.text(0.5, 0.5, "No test lap has finished yet - keep training!", transform=a2.transAxes,
                ha="center", va="center", fontsize=14, color=P.MUTED)
    a2.set_ylabel("lap time (standing start)")
    a2.set_xlabel("practice steps (millions)")
    a2.set_title("Lap times: the yellow band is the gap still left to the benchmark")
    fig.suptitle("Learning curve", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return P.save(fig, os.path.join(ALT, "3_learning_curve.png"))


# ----------------------------------------------------------------- 4 telemetry
def time_gap(lap, bench_tel, L):
    d = mono(arr(lap, "dist"))
    t = arr(lap, "t")
    grid = np.arange(0, min(d[-1], L), 5.0)
    gap = np.interp(grid, d, t) - np.interp(grid, bench_tel["dist"], bench_tel["t"])
    return grid, gap


def pedal_strip(ax, d, throttle, brake, title, edge):
    ax.fill_between(d, throttle * 100, color="#66bb6a", alpha=0.75, lw=0, label="throttle")
    ax.fill_between(d, -brake * 100, color="#ef5350", alpha=0.85, lw=0, label="brake")
    ax.set_ylim(-105, 105)
    ax.set_yticks([-100, 0, 100])
    ax.set_yticklabels(["brake", "0", "throttle"])
    ax.set_title(title, fontsize=10, loc="left", color=edge)
    P.shade_corners(ax, label=False, km=True)


def fig_telemetry(track, laps, bench, best, first):
    b = {k: np.asarray(v) for k, v in bench["standing"].items()}
    fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(15, 12.5), sharex=True,
                                         gridspec_kw={"height_ratios": [2.2, 0.75, 0.75, 1.4]})
    a1.plot(b["dist"] / 1000, b["v"] * KMH, color=P.BENCH, lw=1.6,
            label=f"benchmark {fmt_time(bench['standing_lap'])}")
    a1.plot(arr(best, "dist") / 1000, arr(best, "v") * KMH, color=P.AI, lw=1.6,
            label=f"AI best: {lap_label(best)}")
    a1.set_ylabel("speed (km/h)")
    a1.legend(loc="lower right", fontsize=9)
    P.shade_corners(a1, km=True)

    pedal_strip(a2, b["dist"] / 1000, b["throttle"], b["brake"],
                "Benchmark: throttle and brake", P.BENCH)
    pedal_strip(a3, arr(best, "dist") / 1000, arr(best, "throttle"), arr(best, "brake"),
                "AI best lap: throttle and brake", P.AI)
    a3.legend(loc="lower right", fontsize=9, ncol=2)

    grid, gap = time_gap(best, b, track.length)
    a4.plot(grid / 1000, gap, color=P.AI, lw=1.6)
    a4.fill_between(grid / 1000, gap, color=P.AI, alpha=0.15)
    a4.axhline(0, color=P.BENCH, lw=1, ls="--")
    a4.set_ylabel("seconds behind\nthe benchmark")
    a4.set_xlabel("distance along the lap (km)")
    a4.set_title("Time gap to the benchmark", fontsize=10, loc="left")
    P.shade_corners(a4, label=False, km=True)
    a4.set_xlim(0, track.length / 1000)
    fig.suptitle("Telemetry: AI best lap vs benchmark", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return P.save(fig, "4_telemetry.png")


# ----------------------------------------------------------------- 5 grip
def ai_grip_used(lap):
    v = arr(lap, "v")
    tyre_long = arr(lap, "long_g") + (C.DRAG_K * v * v + C.ROLL) / C.G
    return np.minimum(1.0, np.hypot(arr(lap, "lat_g"), tyre_long) / C.grip_g(v))


def grip_share(lat_g, long_g, v):
    """Sideways and forwards tyre force as a share of the grip available at that speed
    (drag is added back, so only what the tyres do counts). 1.0 = at the limit."""
    g = C.grip_g(v)
    tyre_long = long_g + (C.DRAG_K * v * v + C.ROLL) / C.G
    return lat_g / g, tyre_long / g


def corner_braking(dist, brake, v, lo, hi, prev_hi):
    """(where hard braking starts, slowest speed) for one corner, or None if not reached."""
    seg = np.where((dist >= lo) & (dist <= hi))[0]          # the corner itself
    if len(seg) < 3:
        return None
    i_min = seg[np.argmin(v[seg])]
    # hard braking: searched from the end of the previous corner up to the slowest point
    win = np.where((dist >= max(lo - 600, prev_hi)) & (dist <= dist[i_min]) & (brake > 0.5))[0]
    return (dist[win[0]] if len(win) else None), v[i_min] * KMH


def fig_grip(track, bench, best):
    """Picture 5 (original design): g-g diagram and grip used along the lap."""
    b = {k: np.asarray(v) for k, v in bench["standing"].items()}
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.7])
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    a1.scatter(b["lat_g"], b["long_g"], s=2, color=P.BENCH, alpha=0.35, label="benchmark")
    a1.scatter(arr(best, "lat_g"), arr(best, "long_g"), s=4, color=P.AI, alpha=0.6, label="AI best lap")
    th = np.linspace(0, 2 * np.pi, 200)
    for kmh in (100, 200, 300):
        g = C.grip_g(kmh / KMH)
        a1.plot(g * np.cos(th), g * np.sin(th), color=P.MUTED, lw=0.8, ls="--")
        a1.text(g * 0.71, g * 0.71, f"{kmh} km/h", color=P.MUTED, fontsize=8)
    a1.set_aspect("equal")
    a1.set_xlabel("sideways g (+ = turning left)")
    a1.set_ylabel("forward g (+ accelerating, - braking)")
    a1.set_title("g-g diagram (dashed: grip limit at each speed)", pad=44)
    a1.legend(loc="upper right", markerscale=4)

    a2.plot(b["dist"] / 1000, b["grip_used"] * 100, color=P.BENCH, lw=1, label="benchmark")
    d, used = arr(best, "dist") / 1000, ai_grip_used(best) * 100
    a2.plot(d, used, color=P.AI, lw=0.5, alpha=0.3)
    smooth = np.convolve(used, np.ones(20) / 20, "same")   # 1-second average
    a2.plot(d, smooth, color=P.AI, lw=1.6, label="AI best lap (1 s average; faint = raw)")
    a2.set_ylim(0, 105)
    a2.set_xlim(0, track.length / 1000)
    a2.set_xlabel("distance along the lap (km)")
    a2.set_ylabel("grip used (%)")
    a2.set_title("How much of the tyres' grip is being used", pad=44)
    a2.legend(loc="lower right")
    P.shade_corners(a2, km=True)
    fig.suptitle("Using the grip: a perfect driver sits at 100% in every corner and braking zone",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return P.save(fig, "5_grip_usage.png")


def fig_grip_alt(track, bench, best):
    """Picture 5, alternative design: friction circle in percent + corner-by-corner braking."""
    b = {k: np.asarray(v) for k, v in bench["standing"].items()}
    fig = plt.figure(figsize=(16, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35])
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --- left: friction circle in percent: the circle is the limit at every speed
    bx, by = grip_share(b["lat_g"], b["long_g"], b["v"])
    ax_, ay_ = grip_share(arr(best, "lat_g"), arr(best, "long_g"), arr(best, "v"))
    th = np.linspace(0, 2 * np.pi, 300)
    a1.fill(np.cos(th), np.sin(th), color="white", alpha=0.04)
    a1.plot(np.cos(th), np.sin(th), color="white", lw=2)
    a1.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color=P.MUTED, lw=0.8, ls="--")
    a1.text(0.36, 0.36, "50%", color=P.MUTED, fontsize=9)
    a1.text(0.72, 0.72, "100%\n(the limit)", color="white", fontsize=9)
    a1.scatter(bx, by, s=3, color=P.BENCH, alpha=0.35, lw=0, label="benchmark (perfect driver)")
    a1.scatter(ax_, ay_, s=8, color=P.AI, alpha=0.65, lw=0, label="AI best lap (20 dots per second)")
    a1.set_xlim(-1.4, 1.4); a1.set_ylim(-1.45, 1.35)
    a1.set_aspect("equal")
    a1.axhline(0, color=P.MUTED, lw=0.5); a1.axvline(0, color=P.MUTED, lw=0.5)
    a1.set_xticks([]); a1.set_yticks([])
    for txt, x, y in (("BRAKING", 0, -1.36), ("ACCELERATING", 0, 1.25),
                      ("TURNING\nLEFT", -1.25, 0.13), ("TURNING\nRIGHT", 1.25, 0.13)):
        a1.text(x, y, txt, ha="center", va="center", color=P.MUTED, fontsize=9, fontweight="bold")
    a1.set_title("Share of the tyres' grip used, in every direction\n"
                 "on the circle = at the limit  ·  inside = grip left unused\n"
                 "outside = asking for more than the tyres have: the car slides and scrubs off speed",
                 fontsize=10.5)
    a1.legend(loc="lower left", fontsize=8.5, markerscale=2.5, bbox_to_anchor=(-0.02, -0.12))

    # --- right: per corner, slowest speed (dumbbell) + where braking starts
    names, rows = [], []
    prev_hi = 0
    for name, lo, hi in CORNERS:
        rb = corner_braking(b["dist"], b["brake"], b["v"], lo, hi, prev_hi)
        ra = corner_braking(arr(best, "dist"), arr(best, "brake"), arr(best, "v"), lo, hi, prev_hi)
        prev_hi = hi
        names.append(name); rows.append((rb, ra))
    yy = np.arange(len(names))[::-1]
    for y, (rb, ra) in zip(yy, rows):
        if ra is not None:
            a2.plot([rb[1], ra[1]], [y, y], color=P.MUTED, lw=2.5, zorder=1)
            a2.plot(ra[1], y, "o", color=P.AI, ms=12, zorder=3)
            dv = ra[1] - rb[1]
            a2.text(max(ra[1], rb[1]) + 10, y + 0.14, f"{dv:+.0f} km/h", va="center", fontsize=10,
                    color="white", fontweight="bold")
            if rb[0] is not None and ra[0] is not None:
                dd = rb[0] - ra[0]
                txt = (f"AI brakes {abs(dd):.0f} m {'earlier' if dd > 0 else 'later'}"
                       if abs(dd) >= 5 else "brakes at the same point")
            elif rb[0] is None:
                txt = "benchmark doesn't brake"
            else:
                txt = "AI doesn't brake hard"
            a2.text(max(ra[1], rb[1]) + 10, y - 0.2, txt, va="center", fontsize=8.5, color=P.MUTED)
        a2.plot(rb[1], y, "o", color=P.BENCH, ms=12, zorder=2)
    a2.plot([], [], "o", color=P.BENCH, ms=10, label="benchmark")
    a2.plot([], [], "o", color=P.AI, ms=10, label="AI best lap")
    a2.set_yticks(yy)
    a2.set_yticklabels(names, fontsize=10.5)
    a2.set_xlim(0, 420)
    a2.set_ylim(-0.7, len(names) - 0.3)
    a2.set_xlabel("slowest speed in the corner (km/h)")
    a2.set_title("Corner by corner: slowest speed, and where hard braking starts", fontsize=11)
    a2.legend(loc="lower right", fontsize=9)
    a2.grid(axis="y", visible=False)
    fig.suptitle("Grip and braking: how close to the limit the AI drives",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return P.save(fig, os.path.join(ALT, "5_grip_usage.png"))


# ----------------------------------------------------------------- 6 line vs benchmark
def fig_line_vs_bench(track, bench, best):
    x, y, v = arr(best, "x"), arr(best, "y"), arr(best, "v")
    bx, by = np.array(bench["line_x"] + bench["line_x"][:1]), np.array(bench["line_y"] + bench["line_y"][:1])
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.2, 1.2, 1, 1])
    big = fig.add_subplot(gs[:, :2])
    norm = Normalize(50, 350)
    P.draw_track(big, track)
    big.plot(bx, by, "--", color=P.BENCH, lw=0.8, zorder=3)
    lc = speed_line(big, x, y, v, lw=2.2, norm=norm)
    P.label_corners(big, track, size=9)
    big.set_title(f"AI best line ({lap_label(best)}), coloured by speed")
    for k, name in enumerate(ZOOMS6):
        ax = fig.add_subplot(gs[k // 2, 2 + k % 2])
        P.draw_track(ax, track, lw=1.2)
        ax.plot(bx, by, "--", color=P.BENCH, lw=1.6, zorder=5)
        speed_line(ax, x, y, v, lw=2.6, norm=norm, zorder=4)
        P.zoom_to(ax, track, *CORNER[name], pad=20)
        ax.set_title(name, fontsize=10)
    cb = fig.colorbar(lc, ax=fig.axes, shrink=0.5, pad=0.01)
    cb.set_label("AI speed (km/h)")
    fig.suptitle("The AI's line (coloured) vs the benchmark line (dashed blue)",
                 fontsize=15, fontweight="bold")
    return P.save(fig, "6_line_vs_benchmark.png")


# ----------------------------------------------------------------- summary
def corner_min(dist, v, a, b):
    m = (dist >= a) & (dist <= b)
    return v[m].min() * KMH if m.any() else None


def summary(track, laps, bench, best, first, info):
    b = {k: np.asarray(v) for k, v in bench["standing"].items()}
    out = ["F1 AI at Monza - summary", "=" * 48]
    out.append(f"Training: {info.get('steps', laps[-1]['steps']):,} practice steps, "
               f"{info.get('minutes', laps[-1]['minutes']):.0f} minutes, {len(laps)} test laps")
    nfin = sum(l["finished"] for l in laps)
    out.append(f"Finished test laps: {nfin} of {len(laps)}")
    if first:
        out.append(f"First finished lap: {fmt_time(first['lap_time'])} after "
                   f"{first['steps'] / 1e6:.2f}M steps ({first['minutes']:.0f} min)")
    else:
        out.append("No test lap has finished yet. Figures use the attempt that got furthest:")
        out.append(f"  {lap_label(best)}, {best['reason']} at {best['where']}")
    out.append("")
    out.append(f"Benchmark (maths): {fmt_time(bench['standing_lap'])} from a standing start, "
               f"{fmt_time(bench['flying_lap'])} flying lap")
    if best["finished"]:
        out.append(f"AI best lap:       {fmt_time(best['lap_time'])} after {best['steps'] / 1e6:.2f}M "
                   f"steps ({best['minutes']:.0f} min)")
        gap = best["lap_time"] - bench["standing_lap"]
        out.append(f"Gap to benchmark:  {gap:+.3f} s ({100 * gap / bench['standing_lap']:+.1f}%)")
    v = arr(best, "v")
    out.append(f"Top speed:         AI {v.max() * KMH:.0f} km/h, benchmark {b['v'].max() * KMH:.0f} km/h")
    # weaving: sideways g where the track is straight
    d = arr(best, "dist")
    idx = [track.index_at(s) for s in d]
    straight = np.abs(track.CURV[idx]) < 1 / 1000
    if straight.any():
        weave = np.abs(arr(best, "lat_g"))[straight].mean()
        out.append(f"Weaving on the straights: {weave:.2f} g sideways on average (smooth driver: ~0)")
    out.append("")
    out.append("Slowest point of each corner:   AI       benchmark")
    bd = b["dist"]
    for name, a, c in CORNERS:
        ai_v = corner_min(d, v, a, c)
        bv = corner_min(bd, b["v"], a, c)
        ai_s = f"{ai_v:4.0f}" if ai_v is not None else "   -"
        out.append(f"  {name:14s}            {ai_s} km/h  {bv:4.0f} km/h")
    out.append("")
    out.append("Note: the benchmark is a perfect driver on the minimum-curvature line - a strong")
    out.append("reference, not a proof of the fastest possible lap.")
    text = "\n".join(out)
    with open(os.path.join(paths.RESULTS, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("saved results\\summary.txt\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    files = paths.RunFiles(args.name)
    P.dark_style()
    track = get_track()
    bench = get_optimal()
    P.track_figure(track)
    laps = ai.load_test_laps(files.test_laps)
    if not laps:
        print("No test laps yet - run 2_train first. (Drew the track only.)")
        return
    best = best_lap(laps)
    first = next((l for l in laps if l["finished"]), None)
    info = paths.read_json(files.info, {})
    fig_racing_lines(track, laps, bench)
    fig_learning(track, laps, bench)
    fig_learning_alt(track, laps, bench)
    fig_telemetry(track, laps, bench, best, first)
    fig_grip(track, bench, best)
    fig_grip_alt(track, bench, best)
    fig_line_vs_bench(track, bench, best)
    summary(track, laps, bench, best, first, info)
    if os.name == "nt" and not args.no_open:
        os.startfile(paths.RESULTS)


if __name__ == "__main__":
    main()
