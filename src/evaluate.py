
"""evaluate.py - metric persistence + reporting."""
from __future__ import annotations
import json
from pathlib import Path

def save_metrics(m, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    return m

def print_report(m):
    print("=" * 44); print("Evaluation report"); print("=" * 44)
    for k, v in m.items():
        if isinstance(v, float):
            print("  %-22s: %.4f" % (k, v))
        else:
            print("  %-22s: %s" % (k, v))
