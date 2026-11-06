# Monza racing line AI

An AI that teaches itself to drive a simplified Formula 1 car around Monza using
**reinforcement learning**: nobody shows it how to drive and it never sees a human lap.
It learns purely by trial and error, earning points for every metre it covers and losing
points when it crashes. Over millions of practice steps it discovers braking points,
turn-in and a racing line. A mathematical "perfect driver" lap with the same car is
calculated alongside, so you can see how close your AI gets.

**See it in action:** [watch a trained AI drive Monza live in your browser](https://claude.ai/artifact/7SFGz5HcF4BNRLiZfiTkZP),
with the lap times and results it reached.

Everything you need to train the AI on your own computer is here: the simulator, the
training setup, the game window and the report generator. Your AI starts from zero, and
the programs produce your own lap times, telemetry and pictures as it learns.

## What you need

* A Windows PC (Mac and Linux work too, see below). A processor with 8 or more cores helps:
  training runs 16 cars in parallel.
* **Python 3.12**: install it from python.org, or with `winget install Python.Python.3.12`.
* The libraries: open a terminal in this folder and run
  ```
  py -3.12 -m pip install -r requirements.txt
  ```

On Mac or Linux, run the programs with `python3 1_show_track.py` and so on, instead of the
`.bat` files.

## The four programs (double-click the .bat files)

| | Program | What it does |
|---|---|---|
| 1 | `1_show_track` | Draws Monza with its corners named → `results/1_monza_track.png` |
| 2 | `2_train` | Trains the AI, continuing from your saved one (or starting a new one if there is none). Stop any time with Ctrl+C or by closing the window: everything up to the last test lap is kept. |
| 2 | `2_train_fresh` | Starts a brand-new AI from zero. The previous AI and its test laps are moved to `models/old/` (not deleted). |
| 3 | `3_watch_ai` | A game window where your best AI so far drives a lap, with speed, pedals, g-forces and the live gap to the perfect-driver benchmark. Keys: SPACE pause, R restart, +/- zoom, B benchmark line, 1–4 playback speed, ESC quit. |
| 4 | `4_make_report` | Turns your training history into pictures and numbers in `results/` |

## Training: what to expect

* **It takes hours, not minutes.** Keep the PC otherwise idle, and set it never to sleep
  while training (it only learns while awake).
* Every 50,000 practice steps the AI drives one test lap from a standstill and prints a line:
  ```
   12.35M steps |   80.4 min | LAP 1:33.870                     | best: LAP 1:33.212
  ```
  That's practice steps so far, training time, this test lap's result, and the best so far.
* **What's a practice step?** The AI makes one decision every 1/15 of a second of driving
  (how much to steer, and how hard to press the throttle or brake). Each decision is one
  practice step, so one million steps is about 18.5 hours of simulated driving, shared
  between the 16 cars.
* **The first chicane is the big hurdle.** Expect the first million or more steps to end
  at the Rettifilo, the first chicane, before the first full lap appears. After that,
  lap times fall quickly at first and then more and more slowly.
* Training runs in **two phases automatically**: *exploring* until the AI has finished 20
  test laps, then *fine-tuning*. It prints a line when it switches.
* **Results vary between runs.** The randomness in training (set by a "seed") changes how
  quickly an AI learns. If one seems stuck for a long time, try a fresh one.
* When the best lap stops improving for several million steps, stop: more training after
  that tends to make the latest AI drift, though your best one is always kept safely.

## What you'll get in `results/`

| File | Shows |
|---|---|
| `1_monza_track.png` | The track and its corners |
| `2_racing_lines.png` | Up to 60 of your test laps, evenly spaced through training, coloured from purple (early) to yellow (late), with × where a lap went off |
| `3_learning_curve.png` | How far each test lap got, and the lap times with the best-so-far line and the benchmark |
| `4_telemetry.png` | Speed along the lap for your best lap and the benchmark, both drivers' pedals, and the time gap |
| `5_grip_usage.png` | How much of the tyres' grip your AI uses, in every direction and along the lap |
| `6_line_vs_benchmark.png` | Your AI's best line coloured by speed, against the benchmark line, with six corner close-ups |
| `alternative_versions/` | Different designs of pictures 3 and 5 |
| `summary.txt` | The key numbers: training totals, best lap, gap to the benchmark, top speed, slowest point of each corner |

Progress is shown in practice steps rather than minutes, so a slower computer doesn't
distort the picture. Before any lap has finished, the report uses the attempt that got
furthest.

## How it works (`sim/`)

| File | Role |
|---|---|
| `track.py` | Monza's centre line and widths (5.79 km, a point every ~5 m). Precomputes the edges, curvature and distance along the track, and finds "where is the car?" quickly. |
| `car.py` | The car, simulated 60 times a second: 800 kg, 750 kW, tyre grip that grows with speed (≈2.4 g at 150 km/h, 4.5 g at 300 km/h) and is shared between turning and braking (the friction circle), traction control, air drag. |
| `race.py` | One lap attempt: timing, the rules (off track = all four wheels over the white line; stuck = below 3 m/s after 3 s), telemetry at 20 Hz |
| `f1_env.py` | The AI's world: what it sees, what it controls, and its score |
| `ai.py` | The learning setup (PPO with 16 cars in parallel, the two training phases), test laps, crash-safe saving |
| `optimal.py` | The perfect-driver benchmark |
| `corners.py`, `plotting.py`, `viewer.py`, `paths.py` | Corner names, drawing helpers, the game window, file locations |

* **What the AI sees:** 24 numbers: its speed, steering, position across the track, angle
  to the track, and where the road goes at 10 points from 5 to 290 m ahead.
* **What it controls:** steering ("use this share of the available grip for turning") and
  one pedal (positive = throttle, negative = brake), 15 times a second.
* **Its score:** +1 per 10 m of progress, a small cost for jerky steering, −10 for going off
  track or getting stuck (either one ends the attempt).
* **How it practises:** 75% of attempts start somewhere random on the track at 36–180 km/h,
  25% from a standstill on the start line. Test laps are always from a standstill.
* **The benchmark:** the line through the track that bends least (minimum curvature),
  driven at the limit of grip everywhere. It's a strong reference for a perfect driver,
  but not a proof of the fastest possible lap: a different line can be faster, and an AI
  may find one.

## Lessons learned while building it

* **Normalising the AI's inputs and rewards is essential** (`VecNormalize`). Without it,
  the AI never gets past the first chicane.
* **Stopping must cost as much as crashing.** Otherwise the AI learns to park before hard
  corners: zero points beats −10.
* **No grass run-off practice, and no practice starts at racing speed.** Both made learning
  worse.
* **Steer by grip, not by wheel angle.** A fixed wheel angle that is gentle at 100 km/h is
  far beyond the grip at 300 km/h, and makes the AI weave on the straights.
* **Keep the AI curious early on.** Without an entropy bonus (a small reward for staying
  a bit unpredictable), AIs stopped experimenting and decided crashing into the first
  chicane at full speed was their best option. With it, they got through.
* **Then calm it down.** Once the AI can drive, the same bonus makes its practice too random
  to refine braking points. A smaller bonus and a smaller learning rate let it polish its
  laps. `2_train` switches between the two automatically.
* **Keep the best, not just the latest.** Late in training the AI can get worse. A copy of
  the AI from its best test lap is always kept, and that's the one `3_watch_ai` shows.

## Options for `2_train.py`

```
--fresh        start from zero (the old AI is moved to models/old/)
--seed N       choose the random seed
--name NAME    train a separate AI side by side, e.g. --name test
--envs N       cars practising in parallel (default 16; use 8 each for two AIs at once)
--minutes M    stop by itself after M minutes of training
--ent X        fix the entropy bonus (turns off the automatic switch)
--lr X         fix the learning rate
```

Watch or report on a named AI with `--name NAME` added to `3_watch_ai.py` or
`4_make_report.py`.

## Credits and licence

* The code is under the MIT licence (see `LICENSE`).
* **Track data:** Monza centre line and track widths from the
  [TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database) (Technical
  University of Munich), used under the LGPL-3.0 licence (see `sim/Monza_data_LICENSE.txt`).
* **Libraries:** numpy, scipy, matplotlib, pygame, gymnasium, stable-baselines3 and PyTorch.
* **Inspiration:** Yosh's TrackMania AI videos.
* Built with the help of Claude Code.
