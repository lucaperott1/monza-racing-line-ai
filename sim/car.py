"""A simplified F1 car seen from above, simulated in steps of 1/60 s.

Plain-language version:
* The tyres can push the car with a limited force: the GRIP. Grip grows with speed,
  because the wings press the car into the road (downforce).
* That grip is shared between turning and braking/accelerating (the "friction circle").
  Ask for more than there is and the car runs wide and scrubs off speed.
* The engine has 750 kW, but traction control never lets it spin the wheels.
* Air drag grows with speed squared, so the car has a top speed.
"""
import math

G = 9.81
DT = 1.0 / 60.0

# All car numbers in one place. The benchmark (optimal.py) uses exactly these too,
# and recalculates itself automatically when any of them change.
CAR = {
    "mass": 800.0,             # kg
    "power": 750e3,            # W
    "drag_area": 1.1,          # Cd x frontal area, m^2
    "air_density": 1.225,      # kg/m^3
    "rolling": 0.2,            # m/s^2 rolling resistance
    "wheelbase": 3.6,          # m
    "steer_lock_deg": 20.0,    # max front-wheel angle
    "steer_rate": 3.0,         # full locks per second
    "grip_base": 1.7,          # g at walking pace
    "grip_aero": 0.0004,       # extra g per (m/s)^2 (downforce)
    "scrub": 0.5,              # share of excess demand lost as speed when sliding
}

MASS = CAR["mass"]
POWER = CAR["power"]
DRAG_K = 0.5 * CAR["air_density"] * CAR["drag_area"] / MASS   # drag decel = DRAG_K * v^2
ROLL = CAR["rolling"]
WB = CAR["wheelbase"]
LOCK = math.radians(CAR["steer_lock_deg"])
TAN_LOCK = math.tan(LOCK)
STEER_STEP = CAR["steer_rate"] * LOCK * DT   # how far the wheels can turn in one step
GRIP_BASE = CAR["grip_base"]
GRIP_AERO = CAR["grip_aero"]
SCRUB = CAR["scrub"]


def grip_g(v):
    """Available grip in g at speed v (m/s)."""
    return GRIP_BASE + GRIP_AERO * v * v


def grip_acc(v):
    """Available grip in m/s^2."""
    return G * (GRIP_BASE + GRIP_AERO * v * v)


def action_to_controls(steer_action, pedal_action, v):
    """THE one conversion from the AI's two numbers to the car's controls.
    Used by training, test laps and the viewer alike.

    steer_action in [-1, 1] = how much of the available grip to use for turning
    (+ = left). Converted to a front-wheel angle that, at this speed, asks for that
    share of the grip, but never more than the steering lock allows.
    pedal_action in [-1, 1]: x1.5 then clipped, + = throttle, - = brake."""
    a = -1.0 if steer_action < -1.0 else (1.0 if steer_action > 1.0 else steer_action)
    vv = v if v > 1.0 else 1.0
    max_curv = grip_acc(v) / (vv * vv)
    lock_curv = TAN_LOCK / WB
    curv = a * (max_curv if max_curv < lock_curv else lock_curv)
    wheel = math.atan(curv * WB)
    p = pedal_action * 1.5
    p = -1.0 if p < -1.0 else (1.0 if p > 1.0 else p)
    return wheel, p


class Car:
    __slots__ = ("x", "y", "heading", "v", "steer", "lat_g", "long_g", "grip_used",
                 "throttle", "brake")

    def __init__(self, x, y, heading, v=0.0):
        self.x, self.y, self.heading, self.v = x, y, heading, v
        self.steer = 0.0   # current front-wheel angle (rad)
        self.lat_g = self.long_g = self.grip_used = 0.0
        self.throttle = self.brake = 0.0

    def step(self, wheel_target, pedal):
        # 1. steering wheel moves towards the target at a limited speed
        d = wheel_target - self.steer
        if d > STEER_STEP:
            d = STEER_STEP
        elif d < -STEER_STEP:
            d = -STEER_STEP
        self.steer += d

        v = self.v
        a_grip = G * (GRIP_BASE + GRIP_AERO * v * v)
        curv = math.tan(self.steer) / WB          # path curvature the wheels ask for
        a_lat = v * v * curv                      # sideways acceleration that needs

        # 2. pedal -> forward/backward tyre force
        if pedal >= 0.0:
            engine = POWER / (MASS * (v if v > 1.0 else 1.0))
            if engine > a_grip:                   # traction control
                engine = a_grip
            a_long = pedal * engine
            self.throttle, self.brake = pedal, 0.0
        else:
            a_long = pedal * a_grip               # full brake = all the grip
            self.throttle, self.brake = 0.0, -pedal

        # 3. friction circle: turning + braking/accelerating <= grip
        total = math.sqrt(a_lat * a_lat + a_long * a_long)
        scrub = 0.0
        if total > a_grip:
            k = a_grip / total
            scrub = SCRUB * (total - a_grip)      # sliding costs speed
            a_lat *= k
            a_long *= k
            curv *= k                             # the car runs wide
        self.grip_used = (total if total < a_grip else a_grip) / a_grip

        # 4. drag, rolling resistance, move the car
        acc = a_long - DRAG_K * v * v - scrub - (ROLL if v > 0.0 else 0.0)
        v_new = v + acc * DT
        if v_new < 0.0:
            v_new = 0.0
        vm = 0.5 * (v + v_new)
        self.heading += vm * curv * DT
        self.x += vm * math.cos(self.heading) * DT
        self.y += vm * math.sin(self.heading) * DT
        self.v = v_new
        self.lat_g = a_lat / G
        self.long_g = (v_new - v) / DT / G
