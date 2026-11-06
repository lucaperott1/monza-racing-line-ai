"""Monza: centre line + track widths (TUMFTM racetrack-database).

Everything is precomputed once: distance along the track, direction, curvature
and the two edges. Plain Python lists are kept next to the numpy arrays because
looking up single values in a list is much faster than in a numpy array, and the
car asks "where am I?" every physics step.
"""
import bisect
import math

import numpy as np

from . import paths

OFF_TRACK_MARGIN = 1.0  # car centre this far past the edge = all four wheels over the line


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track:
    def __init__(self, csv_path=paths.TRACK_CSV):
        x, y, wr, wl = [], [], [], []
        with open(csv_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                a, b, c, d = (float(v) for v in line.split(","))
                x.append(a); y.append(b); wr.append(c); wl.append(d)
        n = len(x)
        self.n = n
        self.x, self.y, self.w_right, self.w_left = x, y, wr, wl

        # segments i -> i+1 (the track is a closed loop)
        self.seg_len, self.seg_tx, self.seg_ty, self.seg_heading = [], [], [], []
        for i in range(n):
            j = (i + 1) % n
            dx, dy = x[j] - x[i], y[j] - y[i]
            L = math.hypot(dx, dy)
            self.seg_len.append(L)
            self.seg_tx.append(dx / L)
            self.seg_ty.append(dy / L)
            self.seg_heading.append(math.atan2(dy, dx))
        self.s = [0.0]
        for i in range(n - 1):
            self.s.append(self.s[-1] + self.seg_len[i])
        self.length = self.s[-1] + self.seg_len[-1]

        # direction at each point (average of the two neighbouring segments)
        self.heading, self.nx, self.ny = [], [], []
        for i in range(n):
            h0, h1 = self.seg_heading[i - 1], self.seg_heading[i]
            h = h0 + 0.5 * _wrap(h1 - h0)
            self.heading.append(h)
            self.nx.append(-math.sin(h))  # unit normal pointing to the left
            self.ny.append(math.cos(h))

        # curvature (1/radius, positive = left turn), lightly smoothed
        raw = []
        for i in range(n):
            dh = _wrap(self.seg_heading[i] - self.seg_heading[i - 1])
            raw.append(dh / (0.5 * (self.seg_len[i] + self.seg_len[i - 1])))
        self.curvature = [(raw[i - 1] + 2 * raw[i] + raw[(i + 1) % n]) / 4 for i in range(n)]

        # numpy copies for plotting and maths
        self.X, self.Y = np.array(x), np.array(y)
        self.S = np.array(self.s)
        self.WL, self.WR = np.array(wl), np.array(wr)
        self.NX, self.NY = np.array(self.nx), np.array(self.ny)
        self.HEADING = np.array(self.heading)
        self.CURV = np.array(self.curvature)
        self.left_x, self.left_y = self.X + self.NX * self.WL, self.Y + self.NY * self.WL
        self.right_x, self.right_y = self.X - self.NX * self.WR, self.Y - self.NY * self.WR

    # ------------------------------------------------------------------
    def index_at(self, s):
        """Centre-line index of the segment that contains distance s."""
        s %= self.length
        return bisect.bisect_right(self.s, s) - 1

    def point_at(self, s, hint=None):
        """(x, y, heading, index) of the centre line at distance s.
        With a hint index, walks forward from it instead of searching."""
        s %= self.length
        if hint is None:
            i = self.index_at(s)
        else:
            i = hint
            ss, n = self.s, self.n
            for _ in range(n):
                lo = ss[i]
                hi = ss[i + 1] if i + 1 < n else self.length
                if lo <= s < hi:
                    break
                i = i + 1 if i + 1 < n else 0
        t = s - self.s[i]
        return (self.x[i] + self.seg_tx[i] * t, self.y[i] + self.seg_ty[i] * t,
                self.seg_heading[i], i)

    def locate(self, px, py, hint):
        """Where is the car? Walk forwards/backwards from the previous nearest
        point (hint) instead of searching all 1,159 points.
        Returns (index, distance along track, sideways offset (+ = left),
        left width, right width, track heading)."""
        X, Y, n = self.x, self.y, self.n
        i = hint
        d = (X[i] - px) ** 2 + (Y[i] - py) ** 2
        while True:
            j = i + 1 if i + 1 < n else 0
            dj = (X[j] - px) ** 2 + (Y[j] - py) ** 2
            if dj < d:
                i, d = j, dj
                continue
            k = i - 1 if i > 0 else n - 1
            dk = (X[k] - px) ** 2 + (Y[k] - py) ** 2
            if dk < d:
                i, d = k, dk
                continue
            break
        # project onto the segment after the nearest point, or the one before it
        seg = i
        t = (px - X[i]) * self.seg_tx[i] + (py - Y[i]) * self.seg_ty[i]
        if t < 0:
            seg = i - 1 if i > 0 else n - 1
            t = (px - X[seg]) * self.seg_tx[seg] + (py - Y[seg]) * self.seg_ty[seg]
        L = self.seg_len[seg]
        t = 0.0 if t < 0 else (L if t > L else t)
        tx, ty = self.seg_tx[seg], self.seg_ty[seg]
        lat = tx * (py - Y[seg]) - ty * (px - X[seg])
        f = t / L
        nxt = seg + 1 if seg + 1 < n else 0
        wl = self.w_left[seg] + f * (self.w_left[nxt] - self.w_left[seg])
        wr = self.w_right[seg] + f * (self.w_right[nxt] - self.w_right[seg])
        s = self.s[seg] + t
        return i, s, lat, wl, wr, self.seg_heading[seg]

    def locate_slow(self, px, py):
        """Full search (only for one-off use, e.g. mapping a whole line)."""
        i = int(np.argmin((self.X - px) ** 2 + (self.Y - py) ** 2))
        return self.locate(px, py, i)


_TRACK = None


def get_track():
    global _TRACK
    if _TRACK is None:
        _TRACK = Track()
    return _TRACK
