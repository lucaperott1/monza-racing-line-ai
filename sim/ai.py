"""The AI: a small neural network trained with PPO (stable-baselines3), plus
* the "normaliser" (VecNormalize) that rescales what the AI sees and its rewards to
  a steady range - the single most important ingredient,
* test laps: one lap from a standstill with the AI's best guess (no exploration noise),
* crash-safe saving.
"""
import json
import math
import os
import shutil
import signal
import time

import numpy as np

from . import paths
from .car import action_to_controls
from .corners import corner_at
from .f1_env import PHYSICS_PER_DECISION, F1Env, observe
from .race import Race
from .track import get_track

N_ENVS = 16
TEST_EVERY = 50_000
TEST_MAX_TIME = 300.0          # s; a test lap slower than this is cut off
MAX_GAP_MIN = 5.0              # gaps longer than this (PC asleep) count as 5 min
GAMMA = 0.995
# Entropy bonus = a small reward for staying a bit uncertain, so exploration can't die.
# Phase 1 (explore): 0.01. Without any bonus, three AIs out of three stopped exploring and
#   never got past the Rettifilo.
# Phase 2 (fine-tune), after 20 finished test laps: 0.001. Keeping 0.01 made the pedal
#   wobble explode (12 -> 29) and lap times and reliability got worse.
# The fine-tuning phase also uses a 3x lower learning rate: in a side-by-side test it held
#   ~1:29.7 laps for 4M steps, while the normal rate fell apart within 1M.
ENT_EXPLORE = 0.01
ENT_FINE = 0.001
LR_FINE = 1e-4
FINE_TUNE_AFTER = 20
PPO_SETTINGS = dict(n_steps=256, batch_size=1024, learning_rate=3e-4, gamma=GAMMA,
                    ent_coef=ENT_EXPLORE, policy_kwargs=dict(net_arch=[128, 128]))


# ------------------------------------------------------------------ test laps
def drive_lap(policy, record=True, max_time=TEST_MAX_TIME, on_step=None):
    """One lap from a standstill on the start line under the strict rules.
    policy(obs_list) -> (steer, pedal)."""
    track = get_track()
    race = Race(track, 0.0, 0.0, lap_target=track.length, record=record)
    while not race.done and race.time < max_time:
        a = policy(observe(race))
        for _ in range(PHYSICS_PER_DECISION):
            wheel, pedal = action_to_controls(float(a[0]), float(a[1]), race.car.v)
            race.step(wheel, pedal)
            if on_step:
                on_step(race)
            if race.done:
                break
    if not race.done:
        race.reason = "timeout"
    return race


def lap_record(race, steps, minutes):
    L = race.track.length
    frac = max(0.0, min(1.0, race.progress / L))
    return {
        "steps": int(steps), "minutes": round(minutes, 2),
        "finished": race.reason == "finished",
        "lap_time": round(race.lap_time, 3) if race.lap_time else None,
        "reason": race.reason, "progress": round(race.progress, 1), "frac": round(frac, 4),
        "where": corner_at(race.s), "end_x": round(race.car.x, 1), "end_y": round(race.car.y, 1),
        "tel": race.tel,
    }


def is_better(a, b):
    """Is test lap a better than test lap b? Finishing beats not finishing,
    then faster time, or (if neither finished) further distance."""
    if b is None:
        return True
    if a["finished"] != b["finished"]:
        return a["finished"]
    if a["finished"]:
        return a["lap_time"] < b["lap_time"]
    return a["progress"] > b["progress"]


def describe(lap):
    from .optimal import fmt_time
    if lap is None:
        return "-"
    if lap["finished"]:
        return f"LAP {fmt_time(lap['lap_time'])}"
    word = {"off": "off", "stuck": "stuck", "timeout": "too slow"}.get(lap["reason"], lap["reason"])
    return f"{word} at {lap['where']} ({100 * lap['frac']:.0f}%)"


# ------------------------------------------------------------------ test-lap log
def append_test_lap(path, lap):
    """Add one line to the .jsonl file. Never rewrites the file."""
    line = json.dumps(lap, separators=(",", ":"))
    with open(path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":         # a previous write was cut off mid-line
                f.write(b"\n")
        f.write(line.encode() + b"\n")
        f.flush()
        os.fsync(f.fileno())


def load_test_laps(path):
    laps = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    laps.append(json.loads(line))
                except ValueError:
                    pass                    # half-written line from a crash: skip it
    except OSError:
        pass
    laps.sort(key=lambda d: d["steps"])
    return laps


# ------------------------------------------------------------------ saving
def _swap_in(pairs):
    """os.replace each (tmp, final) with Ctrl+C ignored, so the set is swapped together."""
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for tmp, final in pairs:
            os.replace(tmp, final)
    finally:
        signal.signal(signal.SIGINT, old)


def save_ai(model, venv, model_path, norm_path, info_path, info):
    tmp_model = model_path[:-4] + "_tmp.zip"
    tmp_norm = norm_path + ".tmp"
    tmp_info = info_path + ".tmp"
    model.save(tmp_model)
    venv.save(tmp_norm)
    with open(tmp_info, "w") as f:
        json.dump(info, f, indent=1)
    _swap_in([(tmp_model, model_path), (tmp_norm, norm_path), (tmp_info, info_path)])


def copy_as_best(files, info):
    pairs = []
    for src, dst in ((files.model, files.best_model), (files.norm, files.best_norm)):
        shutil.copyfile(src, dst + ".tmp")
        pairs.append((dst + ".tmp", dst))
    with open(files.best_info + ".tmp", "w") as f:
        json.dump(info, f, indent=1)
    pairs.append((files.best_info + ".tmp", files.best_info))
    _swap_in(pairs)


# ------------------------------------------------------------------ loading a driver
class Driver:
    """A trained AI ready to drive (used by test laps outside training and the viewer)."""

    def __init__(self, model_path, norm_path):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        self.model = PPO.load(model_path, device="cpu")
        self.norm = VecNormalize.load(norm_path, DummyVecEnv([lambda: F1Env()]))
        self.norm.training = False

    def __call__(self, obs):
        o = self.norm.normalize_obs(np.asarray(obs, np.float32)[None, :])
        a, _ = self.model.predict(o, deterministic=True)
        return a[0]


def find_driver(name="", best=True):
    """Paths of the best (or latest) saved AI, or None."""
    f = paths.RunFiles(name)
    for m, n, i in ((f.best_model, f.best_norm, f.best_info), (f.model, f.norm, f.info)):
        if os.path.exists(m) and os.path.exists(n):
            return m, n, paths.read_json(i, {})
        if not best:
            break
    return None
