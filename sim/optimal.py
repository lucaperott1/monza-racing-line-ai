"""The benchmark: what a perfect driver could do with this car on a sensible line.

1. LINE - minimum curvature: at every centre-line point choose a sideways offset so
   the line bends as little as possible overall (sum of squared second differences),
   with the car's centre allowed up to 0.8 m past the white lines.
2. SPEED - with exactly the same car numbers as the simulation:
   * cornering limit: v^2 * curvature = grip(v)
   * forward pass: accelerate as hard as power and the grip left over from cornering allow
   * backward pass: brake as late as the grip allows
   * the speed is the lowest of the three.

Honest note: this is a reference for a perfect driver on the minimum-curvature line.
It is NOT a proof of the fastest possible lap: a different line (e.g. a later apex
that trades corner speed for straight-line speed) can be faster, and the AI is free
to find one.
"""
import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from . import car as C
from . import paths
from .track import get_track

VERSION = 4
EDGE_MARGIN = 0.8   # car centre may go this far past the edge
RESAMPLE = 1.0      # metres between speed-profile points
V_CAP = 150.0       # m/s, "no corner limit"
CURV_SMOOTH = 3     # curvature averaged over 3 points (15 m)


# ----------------------------------------------------------------------------- line
def min_curvature_offsets(track):
    X, Y, NX, NY = track.X, track.Y, track.NX, track.NY
    lo = -(track.WR + EDGE_MARGIN)
    hi = track.WL + EDGE_MARGIN

    def f(a):
        px, py = X + a * NX, Y + a * NY
        rx = np.roll(px, -1) - 2 * px + np.roll(px, 1)
        ry = np.roll(py, -1) - 2 * py + np.roll(py, 1)
        J = np.sum(rx * rx + ry * ry)
        gx = np.roll(rx, 1) - 2 * rx + np.roll(rx, -1)
        gy = np.roll(ry, 1) - 2 * ry + np.roll(ry, -1)
        return J, 2 * (gx * NX + gy * NY)

    res = minimize(f, np.zeros(track.n), jac=True, method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options={"maxiter": 50000, "maxfun": 100000, "ftol": 1e-14, "gtol": 1e-9})
    return res.x


def point_curvature(px, py, smooth=CURV_SMOOTH):
    """Curvature at each 5 m point from the circle through it and its two neighbours,
    averaged over `smooth` points. Smoothing is needed because the measured track
    widths are slightly noisy: where the line hugs an edge, that noise makes tiny
    kinks that a 3.6 m-long car would not feel (a spline fit makes them worse)."""
    ax, ay = px - np.roll(px, 1), py - np.roll(py, 1)
    bx, by = np.roll(px, -1) - px, np.roll(py, -1) - py
    k = 2 * (ax * by - ay * bx) / (np.hypot(ax, ay) * np.hypot(bx, by) * np.hypot(ax + bx, ay + by))
    if smooth > 1:
        w = smooth
        k = np.convolve(np.r_[k[-w:], k, k[:w]], np.ones(w) / w, "same")[w:-w]
    return k


def resample_closed(px, py, step=RESAMPLE):
    """Smooth periodic spline through the line -> points every ~1 m, with curvature."""
    xc, yc = np.append(px, px[0]), np.append(py, py[0])
    u = np.r_[0, np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))]
    sx = CubicSpline(u, xc, bc_type="periodic")
    sy = CubicSpline(u, yc, bc_type="periodic")
    N = int(round(u[-1] / step))
    uu = np.linspace(0, u[-1], N, endpoint=False)
    x, y = sx(uu), sy(uu)
    k5 = point_curvature(px, py)
    kappa = np.interp(uu, u, np.append(k5, k5[0]))
    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)   # ds[i]: point i -> i+1
    return x, y, kappa, ds


# ---------------------------------------------------------------------------- speed
def _grip(v):
    return C.G * (C.GRIP_BASE + C.GRIP_AERO * v * v)


def corner_limit(kappa):
    k = np.abs(kappa)
    denom = k - C.G * C.GRIP_AERO
    v = np.full_like(k, V_CAP)
    m = denom > 1e-9
    v[m] = np.minimum(V_CAP, np.sqrt(C.G * C.GRIP_BASE / denom[m]))
    return v


def _left_over(v, k):
    g = _grip(v)
    lat = v * v * k
    return math.sqrt(max(0.0, g * g - lat * lat))


def braking_limit(vlim, kappa, ds):
    """Backward pass, twice round the loop so the end of the lap 'sees' the start."""
    N = len(vlim)
    v = vlim.copy()
    for _ in range(2):
        for i in range(N - 1, -1, -1):
            j = (i + 1) % N
            vj = v[j]
            dec = _left_over(vj, abs(kappa[j])) + C.DRAG_K * vj * vj + C.ROLL
            cand = math.sqrt(vj * vj + 2 * dec * ds[i])
            if cand < v[i]:
                v[i] = cand
    return v


def _accel(v, k):
    eng = C.POWER / (C.MASS * max(v, 1.0))
    return min(eng, _left_over(v, k)) - C.DRAG_K * v * v - C.ROLL


def forward(vcap, kappa, ds, v_start=None):
    """Forward pass. v_start=None -> flying lap (loop twice), 0 -> standing start."""
    N = len(vcap)
    v = vcap.copy()
    if v_start is None:
        for _ in range(2):
            for i in range(N):
                j = (i + 1) % N
                cand = math.sqrt(max(0.0, v[i] ** 2 + 2 * _accel(v[i], abs(kappa[i])) * ds[i]))
                v[j] = min(vcap[j], cand)
        return np.append(v, v[0])
    out = np.empty(N + 1)
    out[0] = v_start
    for i in range(N):
        j = (i + 1) % N
        cand = math.sqrt(max(0.0, out[i] ** 2 + 2 * _accel(out[i], abs(kappa[i])) * ds[i]))
        out[i + 1] = min(vcap[j], cand)
    return out


def _lap(v, ds):
    vm = 0.5 * (v[:-1] + v[1:])
    dt = ds / np.maximum(vm, 0.1)
    return np.r_[0, np.cumsum(dt)]


def _telemetry(track, x, y, kappa, ds, v):
    """Per-point channels, matching the AI's test-lap telemetry."""
    N = len(x)
    t = _lap(v, ds)
    vv = v[:N]
    acc = (v[1:] ** 2 - v[:-1] ** 2) / (2 * ds)          # m/s^2
    lat = vv * vv * kappa
    tyre_long = acc + C.DRAG_K * vv * vv + C.ROLL          # what the tyres must do
    grip = _grip(vv)
    used = np.minimum(1.0, np.hypot(lat, tyre_long) / grip)
    throttle = np.clip(tyre_long / np.minimum(C.POWER / (C.MASS * np.maximum(vv, 1)), grip), 0, 1)
    brake = np.clip(-tyre_long / grip, 0, 1)
    # distance along the CENTRE line, so it lines up with the AI's telemetry
    dist = np.empty(N)
    i = int(np.argmin((track.X - x[0]) ** 2 + (track.Y - y[0]) ** 2))
    prev, total = None, 0.0
    for k in range(N):
        i, s, *_ = track.locate(float(x[k]), float(y[k]), i)
        if prev is None:
            total = s if s < track.length / 2 else s - track.length
        else:
            d = s - prev
            if d < -track.length / 2:
                d += track.length
            elif d > track.length / 2:
                d -= track.length
            total += d
        prev = s
        dist[k] = total
    r = lambda a, n=2: np.round(a, n).tolist()
    return {"t": r(t[:N], 3), "x": r(x), "y": r(y), "v": r(vv), "dist": r(dist, 1),
            "lat_g": r(lat / C.G), "long_g": r(acc / C.G), "grip_used": r(used, 3),
            "throttle": r(throttle), "brake": r(brake)}


def compute():
    track = get_track()
    print("Calculating the benchmark lap (takes a few seconds)...")
    a = min_curvature_offsets(track)
    lx, ly = track.X + a * track.NX, track.Y + a * track.NY
    x, y, kappa, ds = resample_closed(lx, ly)
    vlim = corner_limit(kappa)
    vbrake = braking_limit(vlim, kappa, ds)
    v_fly = forward(vbrake, kappa, ds)
    v_std = forward(vbrake, kappa, ds, v_start=0.0)
    fly, std = _lap(v_fly, ds)[-1], _lap(v_std, ds)[-1]
    return {
        "version": VERSION, "car": C.CAR,
        "flying_lap": float(fly), "standing_lap": float(std),
        "line_length": float(ds.sum()),
        "offsets": np.round(a, 3).tolist(),
        "line_x": np.round(lx, 2).tolist(), "line_y": np.round(ly, 2).tolist(),
        "flying": _telemetry(track, x, y, kappa, ds, v_fly),
        "standing": _telemetry(track, x, y, kappa, ds, v_std),
    }


def get_optimal():
    """Load the cached benchmark; recalculate if the car numbers changed."""
    data = paths.read_json(paths.OPTIMAL)
    if not data or data.get("version") != VERSION or data.get("car") != C.CAR:
        data = compute()
        paths.write_json_atomic(paths.OPTIMAL, data)
        print(f"Benchmark: flying lap {fmt_time(data['flying_lap'])}, "
              f"standing start {fmt_time(data['standing_lap'])}")
    return data


def fmt_time(t):
    if t is None:
        return "-"
    m = int(t // 60)
    return f"{m}:{t - 60 * m:06.3f}"
