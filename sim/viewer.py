"""The game window: watch a trained AI drive a lap of Monza in real time.

Keys:  SPACE pause | R restart | + / - zoom | B benchmark line on/off
       1 2 3 4 playback speed x1 x2 x4 x8 | ESC quit
"""
import math

import numpy as np
import pygame

from .ai import Driver, find_driver
from .car import action_to_controls
from .corners import corner_at
from .f1_env import PHYSICS_PER_DECISION, observe
from .optimal import fmt_time, get_optimal
from .race import Race
from .track import get_track

W, H = 1280, 800
BG = (18, 20, 24)
GRASS = (24, 38, 28)
TARMAC = (58, 62, 72)
EDGE = (225, 225, 225)
TEXT = (230, 230, 230)
MUTED = (140, 145, 155)
BENCH = (79, 195, 247)
AI_COL = (255, 112, 67)


def speed_colour(v):
    """Blue (slow) -> yellow -> red (fast)."""
    f = max(0.0, min(1.0, v / 95.0))
    if f < 0.5:
        g = f / 0.5
        return (int(60 + 195 * g), int(90 + 150 * g), int(220 - 170 * g))
    g = (f - 0.5) / 0.5
    return (255, int(240 - 190 * g), int(50 - 20 * g))


class Viewer:
    def __init__(self, name=""):
        found = find_driver(name, best=True)
        if found is None:
            raise SystemExit("No trained AI found yet. Run 2_train first.")
        model_path, norm_path, self.info = found
        self.driver = Driver(model_path, norm_path)
        self.track = get_track()
        bench = get_optimal()
        self.bench_xy = list(zip(bench["line_x"], bench["line_y"]))
        st = bench["standing"]
        self.bench_dist, self.bench_t = np.array(st["dist"]), np.array(st["t"])
        self.bench_lap = bench["standing_lap"]

        pygame.init()
        pygame.display.set_caption("F1 AI - Monza")
        self.screen = pygame.display.set_mode((W, H))
        self.font = pygame.font.SysFont("consolas", 18)
        self.big = pygame.font.SysFont("consolas", 34, bold=True)
        self.small = pygame.font.SysFont("consolas", 14)
        self.clock = pygame.time.Clock()
        self.zoom = 2.2          # pixels per metre
        self.show_bench = True
        self.speedup = 1
        self._minimap()
        self.restart()

    # ------------------------------------------------------------------ setup
    def _minimap(self):
        t = self.track
        box = 240
        xmin, xmax, ymin, ymax = t.X.min(), t.X.max(), t.Y.min(), t.Y.max()
        self.mm_scale = box / max(xmax - xmin, ymax - ymin)
        self.mm_origin = (W - 20 - (xmax - xmin) * self.mm_scale, 20)
        self.mm_x0, self.mm_y1 = xmin, ymax
        self.mm_pts = [self._mm(x, y) for x, y in zip(t.x, t.y)]

    def _mm(self, x, y):
        ox, oy = self.mm_origin
        return (ox + (x - self.mm_x0) * self.mm_scale, oy + (self.mm_y1 - y) * self.mm_scale)

    def restart(self):
        t = self.track
        self.race = Race(t, 0.0, 0.0, lap_target=t.length)
        self.trail = []
        self.step_count = 0
        self.action = (0.0, 0.0)
        self.paused = False
        self.finished_at = None

    # ------------------------------------------------------------------ sim
    def advance(self):
        r = self.race
        if r.done:
            return
        if self.step_count % PHYSICS_PER_DECISION == 0:
            self.action = self.driver(observe(r))
        wheel, pedal = action_to_controls(float(self.action[0]), float(self.action[1]), r.car.v)
        r.step(wheel, pedal)
        self.step_count += 1
        if self.step_count % 3 == 0:
            self.trail.append((r.car.x, r.car.y, r.car.v))
        if not r.done and r.time > 300:
            r.done, r.reason = True, "timeout"

    # ------------------------------------------------------------------ drawing
    def to_screen(self, x, y):
        c = self.race.car
        return (W * 0.4 + (x - c.x) * self.zoom, H * 0.5 - (y - c.y) * self.zoom)

    def draw_world(self):
        s = self.screen
        s.fill(GRASS)
        t, c = self.track, self.race.car
        reach = (W / self.zoom) ** 2
        L, R = [], []
        for i in range(t.n):
            near = (t.x[i] - c.x) ** 2 + (t.y[i] - c.y) ** 2 < reach
            L.append(self.to_screen(t.left_x[i], t.left_y[i]) if near else None)
            R.append(self.to_screen(t.right_x[i], t.right_y[i]) if near else None)
        for i in range(t.n):
            j = (i + 1) % t.n
            if L[i] and L[j]:
                pygame.draw.polygon(s, TARMAC, (L[i], L[j], R[j], R[i]))
        for i in range(t.n):
            j = (i + 1) % t.n
            if L[i] and L[j]:
                pygame.draw.line(s, EDGE, L[i], L[j], 2)
                pygame.draw.line(s, EDGE, R[i], R[j], 2)
        # start line
        if L[0] and R[0]:
            pygame.draw.line(s, (255, 80, 80), L[0], R[0], 4)
        if self.show_bench:
            pts = [self.to_screen(x, y) for x, y in self.bench_xy
                   if (x - c.x) ** 2 + (y - c.y) ** 2 < reach]
            for k in range(0, len(pts) - 1, 2):
                if math.dist(pts[k], pts[k + 1]) < 40:
                    pygame.draw.line(s, BENCH, pts[k], pts[k + 1], 2)
        # trail coloured by speed
        prev = None
        for x, y, v in self.trail[-1500:]:
            p = self.to_screen(x, y)
            if prev:
                pygame.draw.line(s, speed_colour(v), prev, p, 3)
            prev = p
        self.draw_car()

    def draw_car(self):
        c = self.race.car
        h = c.heading
        ch, sh = math.cos(h), math.sin(h)
        def pt(f, l):
            return self.to_screen(c.x + ch * f - sh * l, c.y + sh * f + ch * l)
        body = [pt(2.8, 0.0), pt(1.8, 0.45), pt(-1.8, 0.7), pt(-2.4, 0.95), pt(-2.4, -0.95),
                pt(-1.8, -0.7), pt(1.8, -0.45)]
        pygame.draw.polygon(self.screen, AI_COL, body)
        for f, l in ((1.8, 0.85), (1.8, -0.85), (-1.8, 0.9), (-1.8, -0.9)):
            pygame.draw.circle(self.screen, (20, 20, 20), pt(f, l), max(2, int(0.35 * self.zoom)))

    def text(self, msg, pos, font=None, col=TEXT):
        self.screen.blit((font or self.font).render(msg, True, col), pos)

    def bar(self, x, y, w, h, frac, col, label):
        pygame.draw.rect(self.screen, (50, 54, 62), (x, y, w, h), border_radius=3)
        pygame.draw.rect(self.screen, col, (x, y, int(w * max(0, min(1, frac))), h), border_radius=3)
        self.text(label, (x - 90, y - 2), self.small, MUTED)

    def draw_hud(self):
        s, r, c = self.screen, self.race, self.race.car
        panel = pygame.Surface((330, 320), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 210))
        s.blit(panel, (16, 16))
        self.text(f"{c.v * 3.6:5.0f} km/h", (30, 26), self.big)
        self.text(f"lap time  {fmt_time(r.time)}", (30, 74))
        frac = max(0.0, r.progress / self.track.length)
        self.text(f"lap done  {100 * frac:5.1f} %", (30, 98))
        self.text(f"at        {corner_at(r.s % self.track.length)}", (30, 122))
        bt = float(np.interp(r.progress, self.bench_dist, self.bench_t))
        gap = r.time - bt
        self.text(f"vs bench  {gap:+6.2f} s", (30, 146), col=(255, 140, 120) if gap > 0 else (140, 255, 160))
        self.bar(130, 182, 190, 14, c.throttle, (90, 220, 110), "throttle")
        self.bar(130, 204, 190, 14, c.brake, (240, 80, 80), "brake")
        # steering: centred bar
        pygame.draw.rect(s, (50, 54, 62), (130, 226, 190, 14), border_radius=3)
        st = c.steer / math.radians(20)
        mid = 130 + 95
        pygame.draw.rect(s, (255, 200, 80), (min(mid, mid - st * 95), 226, abs(st * 95) + 2, 14))
        self.text("steering", (40, 224), self.small, MUTED)
        self.text(f"g-force   {c.lat_g:+5.1f} side {c.long_g:+5.1f} fwd", (30, 252), self.small)
        self.text(f"grip used {100 * c.grip_used:4.0f} %", (30, 272), self.small)
        self.text(f"x{self.speedup} speed" + ("   PAUSED" if self.paused else ""), (30, 300), self.small, MUTED)

        # who is driving
        info = self.info or {}
        steps = info.get("steps")
        lap = info.get("lap_time")
        best = fmt_time(lap) if lap else (f"got {100 * info.get('frac', 0):.0f}% round" if info else "?")
        line1 = (f"Best AI so far: {steps / 1e6:.2f}M practice steps "
                 f"({info.get('minutes', 0):.0f} min of training)") if steps else "AI"
        line2 = f"its best test lap: {best}   |   benchmark (perfect driver): {fmt_time(self.bench_lap)}"
        panel2 = pygame.Surface((W, 56), pygame.SRCALPHA)
        panel2.fill((10, 12, 16, 210))
        s.blit(panel2, (0, H - 56))
        self.text(line1, (16, H - 50))
        self.text(line2, (16, H - 28), col=MUTED)
        for k, msg in enumerate(("SPACE pause   R restart   +/- zoom",
                                 "B benchmark line   1-4 speed   ESC quit")):
            self.text(msg, (W - 330, 290 + 18 * k), self.small, MUTED)

        # minimap
        pygame.draw.lines(s, (120, 125, 135), True, self.mm_pts, 2)
        pygame.draw.circle(s, AI_COL, self._mm(c.x, c.y), 5)

        if r.done:
            msg = {"finished": f"LAP FINISHED  {fmt_time(r.lap_time)}",
                   "off": f"OFF TRACK at {corner_at(r.s % self.track.length)}",
                   "stuck": "STOPPED", "timeout": "TOO SLOW"}[r.reason]
            img = self.big.render(msg, True, (255, 255, 255))
            box = img.get_rect(center=(W * 0.4, 90))
            pygame.draw.rect(s, (0, 0, 0), box.inflate(30, 20), border_radius=8)
            s.blit(img, box)
            self.text("press R to watch again", (box.x, box.bottom + 14), col=MUTED)

    # ------------------------------------------------------------------ loop
    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key == pygame.K_r:
                        self.restart()
                    elif e.key == pygame.K_b:
                        self.show_bench = not self.show_bench
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.zoom = min(12.0, self.zoom * 1.25)
                    elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.zoom = max(0.3, self.zoom / 1.25)
                    elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        self.speedup = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 4, pygame.K_4: 8}[e.key]
            if not self.paused:
                for _ in range(self.speedup):
                    self.advance()
            self.draw_world()
            self.draw_hud()
            pygame.display.flip()
            self.clock.tick(60)


def run(name=""):
    Viewer(name).run()
