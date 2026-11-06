"""2 - Train the AI. Continues the saved AI (2_train_fresh starts from zero).

Stop any time with Ctrl+C or by closing the window: everything up to the last test
lap is saved safely. Leave the PC otherwise idle and awake while it trains.

Options (for advanced use):
  --fresh        start from zero (old AI and test laps are moved to an 'old' folder)
  --name NAME    train a separate AI, e.g. --name seed2, side by side with the main one
  --seed N       random seed (different seeds learn differently)
  --envs N       cars practising in parallel (default 16)
  --minutes M    stop by itself after M minutes of training
"""
import argparse
import os
import random
import shutil
import sys
import time


def archive_old(files):
    """Move an old AI and its test laps aside instead of deleting them."""
    from sim import paths
    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(paths.MODELS, "old", f"{stamp}{'_' + files.name if files.name else ''}")
    moved = []
    for p in (files.model, files.norm, files.info, files.best_model, files.best_norm,
              files.best_info, files.test_laps):
        if os.path.exists(p):
            os.makedirs(dest, exist_ok=True)
            shutil.move(p, os.path.join(dest, os.path.basename(p)))
            moved.append(os.path.basename(p))
    if moved:
        print(f"Moved the previous AI and its test laps to {os.path.relpath(dest, paths.ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate: how big each learning adjustment is (default 3e-4)")
    ap.add_argument("--ent", type=float, default=None,
                    help="entropy bonus: reward for staying a bit uncertain, so exploration can't die out")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import get_schedule_fn
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from sim import ai, paths
    from sim.optimal import fmt_time, get_optimal

    files = paths.RunFiles(args.name)
    bench = get_optimal()
    if args.fresh:
        archive_old(files)

    info = paths.read_json(files.info) if os.path.exists(files.model) else None
    seed = args.seed if args.seed is not None else (info or {}).get("seed", random.randint(0, 9999))
    n_envs = args.envs or ai.N_ENVS
    venv = SubprocVecEnv([make_env(seed * 100 + i) for i in range(n_envs)])

    if info and os.path.exists(files.norm):
        venv = VecNormalize.load(files.norm, venv)
        venv.training, venv.norm_reward = True, True
        custom = {}
        if args.lr is not None:
            custom = {"learning_rate": args.lr, "lr_schedule": lambda _: args.lr}
        model = PPO.load(files.model, env=venv, device="cpu", custom_objects=custom)
        if args.ent is not None:
            model.ent_coef = args.ent
        print(f"Continuing the saved AI: {info['steps']:,} practice steps, "
              f"{info['minutes']:.0f} min of training so far.")
    else:
        info = {"steps": 0, "minutes": 0.0, "seed": seed, "best": None}
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=ai.GAMMA)
        settings = dict(ai.PPO_SETTINGS)
        if args.ent is not None:
            settings["ent_coef"] = args.ent
        if args.lr is not None:
            settings["learning_rate"] = args.lr
        model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **settings)
        print(f"Starting a brand-new AI from zero (seed {seed}).")
    laps = ai.load_test_laps(files.test_laps)
    best = None
    for lap in laps:
        if ai.is_better(lap, best):
            best = lap
    n_finished = sum(lap["finished"] for lap in laps)
    auto_ent = args.ent is None
    def fine_tune(m):
        m.ent_coef = ai.ENT_FINE
        if args.lr is None:
            m.learning_rate = ai.LR_FINE
            m.lr_schedule = get_schedule_fn(ai.LR_FINE)

    if auto_ent:   # explore until it can finish laps, then fine-tune
        if n_finished >= ai.FINE_TUNE_AFTER:
            fine_tune(model)
        else:
            model.ent_coef = ai.ENT_EXPLORE
    info["ent_coef"] = float(model.ent_coef)
    info["learning_rate"] = float(model.lr_schedule(1.0))
    print(f"Learning rate: {info['learning_rate']:g}")
    phase = "fine-tuning" if model.ent_coef <= ai.ENT_FINE else "exploring"
    print(f"Entropy bonus: {model.ent_coef} ({phase}"
          f"{'' if not auto_ent else ', automatic: switches to fine-tuning after %d finished laps' % ai.FINE_TUNE_AFTER})")
    print(f"Benchmark (perfect driver, standing start): {fmt_time(bench['standing_lap'])}")
    print(f"{n_envs} cars practising in parallel. A test lap every {ai.TEST_EVERY:,} steps.")
    print("Press Ctrl+C (or close this window) to stop. Progress up to the last test lap is kept.\n")

    class TestLaps(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_test = (model.num_timesteps // ai.TEST_EVERY + 1) * ai.TEST_EVERY
            self.last = time.time()
            self.start_minutes = info["minutes"]

        def _on_step(self):
            nonlocal best, n_finished
            if self.num_timesteps < self.next_test:
                return True
            self.next_test += ai.TEST_EVERY
            now = time.time()
            info["minutes"] += min(now - self.last, ai.MAX_GAP_MIN * 60) / 60.0
            self.last = now
            info["steps"] = int(self.num_timesteps)

            norm = self.training_env          # the VecNormalize wrapper
            def policy(obs):
                import numpy as np
                o = norm.normalize_obs(np.asarray(obs, np.float32)[None, :])
                return self.model.predict(o, deterministic=True)[0][0]
            race = ai.drive_lap(policy)
            lap = ai.lap_record(race, info["steps"], info["minutes"])
            new_best = ai.is_better(lap, best)
            n_finished += lap["finished"]
            if auto_ent and lap["finished"] and n_finished == ai.FINE_TUNE_AFTER:
                fine_tune(self.model)
                info["ent_coef"] = float(self.model.ent_coef)
                info["learning_rate"] = float(self.model.lr_schedule(1.0))
                print(f"--- {ai.FINE_TUNE_AFTER} laps finished: switching to fine-tuning "
                      f"(entropy bonus {ai.ENT_EXPLORE} -> {ai.ENT_FINE}, "
                      f"learning rate -> {info['learning_rate']:g}) ---")
            if new_best:
                best = lap
                info["best"] = {k: lap[k] for k in ("steps", "minutes", "finished", "lap_time",
                                                    "frac", "reason", "where")}
            ai.save_ai(self.model, norm, files.model, files.norm, files.info, info)
            ai.append_test_lap(files.test_laps, lap)
            if new_best:
                ai.copy_as_best(files, info["best"])
            mark = "  <- new best" if new_best else ""
            print(f"{info['steps'] / 1e6:6.2f}M steps | {info['minutes']:6.1f} min | "
                  f"{ai.describe(lap):32s} | best: {ai.describe(best)}{mark}", flush=True)
            if args.minutes and info["minutes"] - self.start_minutes >= args.minutes:
                return False
            return True

    try:
        model.learn(total_timesteps=10**12, callback=TestLaps(), reset_num_timesteps=False)
        print("\nTime limit reached - stopped.")
    except KeyboardInterrupt:
        print("\nStopped. Everything up to the last test lap is saved.")
    finally:
        try:
            venv.close()
        except Exception:
            pass


def make_env(seed):
    def _f():
        import torch
        torch.set_num_threads(1)
        from sim.f1_env import F1Env
        return F1Env(seed=seed)
    return _f


if __name__ == "__main__":
    main()
