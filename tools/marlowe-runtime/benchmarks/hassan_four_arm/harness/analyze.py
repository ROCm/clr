"""Frozen arithmetic means and paired intervals; retain every invalid group."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ORDERS = ["ABDC", "BCAD", "CDBA", "DACB"] * 2


def interval(values, confidence):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(
        t.ppf((1 + confidence) / 2, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return dict(
        mean_ms=mean,
        lower_ms=mean - half,
        upper_ms=mean + half,
        confidence=confidence,
        df=len(values) - 1,
        paired_values_ms=values.tolist(),
    )


def group(path):
    status = json.loads((path / "status.json").read_text())
    if not status["valid"]:
        raise ValueError(f"Invalid group {path}: {status.get('error')}")
    raw = json.loads((path / "cadence.json").read_text())
    samples = np.asarray(raw["samples_seconds"]) * 1000
    mean = float(samples.mean())
    if mean != status["cadence_ms"]["mean"]:
        raise ValueError(f"Raw arithmetic mean mismatch: {path}")
    bench = json.loads((path / "measure/bench.json").read_text())
    requests = json.loads((path / "measure/requests.json").read_text())
    if len(requests) != 83 or bench["completed"] != 83 or bench["duration"] < 60:
        raise ValueError(f"Invalid measured request population: {path}")
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    return dict(
        path=str(path),
        mean_ms=mean,
        cadence=status["cadence_ms"],
        request_sha256=digest,
        output_lens=bench["output_lens"],
        client_duration_s=bench["duration"],
        output_tokens=bench["total_output_tokens"],
        mean_accept_length=status["mean_accept_length"],
        acceptance_histogram=status["acceptance_histogram"],
        graph_cadence_count=len(samples),
        output_text_sha256=hashlib.sha256(
            json.dumps(bench["generated_texts"]).encode()
        ).hexdigest(),
    )


def calibrate(root):
    groups = [group(root / f"aa-{i:02d}-A") for i in range(8)]
    if len({g["request_sha256"] for g in groups}) != 1:
        raise ValueError("A/A groups did not draw identical requests")
    means = [g["mean_ms"] for g in groups]
    differences = [means[i + 1] - means[i] for i in range(0, 8, 2)]
    mean = float(np.mean(means))
    delta = max(0.005, 0.005 * mean, 2 * max(abs(d) for d in differences))
    result = dict(
        groups=groups,
        mean_AA_ms=mean,
        pair_differences_ms=differences,
        delta_ms=delta,
        delta_percent=100 * delta / mean,
        resolved=delta <= 0.02 * mean,
        descriptive_only=delta > 0.02 * mean,
        noise_gate="passed" if delta <= 0.02 * mean else "unresolved_high_noise",
        frozen_orders=ORDERS,
    )
    (root / "calibration.json").write_text(json.dumps(result, indent=2))
    (root / "calibration.sha256").write_text(
        hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest()
        + "  calibration.json\n"
    )
    return result


def analyze(root):
    calibration = json.loads((root / "calibration.json").read_text())
    expected = (root / "calibration.sha256").read_text().split()[0]
    if hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen calibration was modified")
    arms = {arm: [] for arm in "ABCD"}
    invalid = []
    complete = []
    for block, order in enumerate(ORDERS):
        rows = {}
        for arm in order:
            path = root / f"block-{block:02d}-{arm}"
            try:
                rows[arm] = group(path)
                arms[arm].append(rows[arm])
            except Exception as error:
                invalid.append(dict(path=str(path), error=repr(error)))
        if len(rows) == 4:
            complete.append(rows)
    result = dict(
        calibration=calibration,
        complete_blocks=len(complete),
        invalid=invalid,
        arm_groups=arms,
        arm_means_ms={
            arm: float(np.mean([g["mean_ms"] for g in rows])) if rows else None
            for arm, rows in arms.items()
        },
        comparisons={},
        accepted=False,
    )
    result["descriptive_only"] = calibration["descriptive_only"]
    if len(complete) == 8:
        if len({row[arm]["request_sha256"] for row in complete for arm in "ABCD"}) != 1:
            raise ValueError("Treatment request draw changed")
        values = {
            arm: np.asarray([row[arm]["mean_ms"] for row in complete]) for arm in "ABCD"
        }
        a, b, c, d = [values[arm] for arm in "ABCD"]
        comparisons = {
            "C-D": interval(c - d, 1 - 0.05 / 3),
            "B-D": interval(b - d, 1 - 0.05 / 3),
            "(B-A)-(D-C)": interval((b - a) - (d - c), 1 - 0.05 / 3),
            "A-D": interval(a - d, 0.95),
        }
        for name, control in (("C-D", c), ("B-D", b), ("A-D", a)):
            comparisons[name]["percent_of_named_control_mean"] = (
                100 * comparisons[name]["mean_ms"] / float(control.mean())
            )
        for row in comparisons.values():
            row["clears_delta_numerically"] = row["lower_ms"] > calibration["delta_ms"]
            row["acceptance_eligible_noise_gate"] = calibration["resolved"]
        result["comparisons"] = comparisons
        result["requires_correctness_engagement_work_equivalence_verdict"] = True
    (root / "results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    result = calibrate(args.root) if args.calibrate else analyze(args.root)
    print(json.dumps(result, indent=2))
