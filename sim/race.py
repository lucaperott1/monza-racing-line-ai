"""One lap attempt: moves the car, keeps track of where it is, applies the rules
(off track, stuck, lap finished) and can record telemetry at 20 Hz."""
import math

from .car import Car, DT, G
from .track import OFF_TRACK_MARGIN

STUCK_SPEED = 3.0   # m/s
STUCK_AFTER = 3.0   # s
RECORD_EVERY = 3    # physics steps -> 20 Hz

CHANNELS = ("t", "x", "y", "v", "throttle", "brake", "steer", "lat_g", "long_g", "dist")


class Race:
    def __init__(self, track, s0=0.0, v0=0.0, lateral=0.0, lap_target=None, record=False):
        """Start at distance s0 along the track, at speed v0, pointing along it.
        lap_target: progress (m) at which the lap counts as finished (None = never)."""
        self.track = track
        px, py, heading, idx = track.point_at(s0)
        px += -math.sin(heading) * lateral
        py += math.cos(heading) * lateral
        self.car = Car(px, py, heading, v0)
        self.idx = idx
        self.idx, self.s, self.lat, self.wl, self.wr, self.track_heading = track.locate(px, py, idx)
        self.progress = 0.0
        self.time = 0.0
        self.steps = 0
        self.lap_target = lap_target
        self.done = False
        self.reason = None       # "finished", "off", "stuck"
        self.lap_time = None
        self.record = record
        self.tel = {k: [] for k in CHANNELS} if record else None
        if record:
            self._record()

    def step(self, wheel, pedal):
        """One physics step (1/60 s). Returns metres of progress made."""
        car, tr = self.car, self.track
        car.step(wheel, pedal)
        self.time += DT
        self.steps += 1
        self.idx, s, self.lat, self.wl, self.wr, self.track_heading = tr.locate(car.x, car.y, self.idx)
        d = s - self.s
        L = tr.length
        if d < -0.5 * L:
            d += L
        elif d > 0.5 * L:
            d -= L
        self.s = s
        before = self.progress
        self.progress += d

        if self.lap_target is not None and self.progress >= self.lap_target:
            # interpolate the moment the line was crossed
            frac = (self.lap_target - before) / d if d > 0 else 1.0
            self.lap_time = self.time - DT * (1.0 - frac)
            self.done, self.reason = True, "finished"
        elif self.lat > self.wl + OFF_TRACK_MARGIN or self.lat < -(self.wr + OFF_TRACK_MARGIN):
            self.done, self.reason = True, "off"
        elif self.time > STUCK_AFTER and car.v < STUCK_SPEED:
            self.done, self.reason = True, "stuck"

        if self.record and (self.steps % RECORD_EVERY == 0 or self.done):
            self._record()
        return d

    def _record(self):
        c, t = self.car, self.tel
        t["t"].append(round(self.time, 3))
        t["x"].append(round(c.x, 2))
        t["y"].append(round(c.y, 2))
        t["v"].append(round(c.v, 2))
        t["throttle"].append(round(c.throttle, 2))
        t["brake"].append(round(c.brake, 2))
        t["steer"].append(round(math.degrees(c.steer), 2))
        t["lat_g"].append(round(c.lat_g, 2))
        t["long_g"].append(round(c.long_g, 2))
        t["dist"].append(round(self.progress, 1))
