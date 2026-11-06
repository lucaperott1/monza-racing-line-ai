"""Where every file lives, plus a crash-safe way to write files."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(ROOT, "runs")
MODELS = os.path.join(ROOT, "models")
TRACK_CSV = os.path.join(SIM, "Monza.csv")
OPTIMAL = os.path.join(RESULTS, "optimal_lap.json")

for _d in (RESULTS, RUNS, MODELS):
    os.makedirs(_d, exist_ok=True)


class RunFiles:
    """File names for one AI. The default AI has no name; extra AIs trained
    side by side (different seeds) get their own name, e.g. 'seed2'."""

    def __init__(self, name=""):
        tag = f"_{name}" if name else ""
        self.name = name
        self.test_laps = os.path.join(RUNS, f"ai_test_laps{tag}.jsonl")
        self.model = os.path.join(MODELS, f"ai_driver{tag}.zip")
        self.norm = os.path.join(MODELS, f"ai_driver{tag}_normalizer.pkl")
        self.info = os.path.join(MODELS, f"ai_driver{tag}.json")
        self.best_model = os.path.join(MODELS, f"best_ai_driver{tag}.zip")
        self.best_norm = os.path.join(MODELS, f"best_ai_driver{tag}_normalizer.pkl")
        self.best_info = os.path.join(MODELS, f"best_ai_driver{tag}.json")


def write_json_atomic(path, data):
    """Write to a temporary file, then swap it in. A crash leaves either the
    old file or the new one, never half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default
