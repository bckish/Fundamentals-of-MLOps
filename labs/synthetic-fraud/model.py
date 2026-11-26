"""Deterministic educational classifier using only synthetic records."""
import hashlib
import json
import math
import random
from pathlib import Path

FEATURES = ("normalized_amount", "velocity", "distance")


def dataset(seed=42, count=800):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        x = [rng.random() for _ in FEATURES]
        label = int(0.8*x[0] + 1.4*x[1] + 1.1*x[2] + rng.uniform(-0.12, 0.12) > 1.7)
        rows.append((x, label))
    return rows


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-40, min(40, value))))


def score(model, x):
    if not isinstance(x, list) or len(x) != len(FEATURES) or any(type(v) not in (int, float) or not 0 <= v <= 1 or not math.isfinite(v) for v in x):
        raise ValueError("features must be three finite numbers in [0,1]")
    return sigmoid(model["bias"] + sum(w*v for w, v in zip(model["weights"], x)))


def train(seed=42):
    rows = dataset(seed)
    dev, heldout = rows[:640], rows[640:]
    model = {"weights": [0.0]*3, "bias": 0.0, "features": list(FEATURES), "seed": seed, "model_version": "synthetic-logistic-v1"}
    for _ in range(500):
        grad = [0.0]*3
        bias = 0.0
        for x, y in dev:
            error = score(model, x) - y
            bias += error
            for i, v in enumerate(x):
                grad[i] += error*v
        model["weights"] = [w - 0.8*g/len(dev) for w, g in zip(model["weights"], grad)]
        model["bias"] -= 0.8*bias/len(dev)
    predictions = [(int(score(model, x) >= 0.5), y) for x, y in heldout]
    tp = sum(p == y == 1 for p, y in predictions)
    fp = sum(p == 1 and y == 0 for p, y in predictions)
    fn = sum(p == 0 and y == 1 for p, y in predictions)
    model["metrics"] = {"heldout_rows": len(heldout), "accuracy": sum(p == y for p, y in predictions)/len(heldout), "precision": tp/(tp+fp) if tp+fp else 0, "recall": tp/(tp+fn) if tp+fn else 0}
    model["dataset_sha256"] = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return model


if __name__ == "__main__":
    model = train()
    Path("model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(model["metrics"], sort_keys=True))
