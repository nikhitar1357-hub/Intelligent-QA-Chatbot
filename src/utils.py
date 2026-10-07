"""
utils.py - small helper functions shared by every module.
"""
import json
import time
from pathlib import Path

# Project root = the folder that contains app.py (one level above /src)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOC_DIR = DATA_DIR / "documents"
KB_PATH = DATA_DIR / "knowledge_base.json"
QUESTIONS_PATH = DATA_DIR / "sample_questions.csv"
RESULTS_DIR = ROOT_DIR / "results"


def load_json(path):
    """Read a JSON file and return the Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Context manager that measures elapsed time in seconds.

    Usage:
        with Timer() as t:
            do_something()
        print(t.elapsed)
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def clip01(x):
    """Force a number into the range [0, 1] (used for confidence scores)."""
    return max(0.0, min(1.0, float(x)))
