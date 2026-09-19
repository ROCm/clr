"""Descriptive reduction of the fixed campaign with retained invalid block00B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

BASE_SHA = "313d02f35d50904bfc8aca944128f3ea981112ed10d77678c5a41b4f5367ef27"
CALIBRATION_SHA = "8e2f896155283350918793a317b840a0faa81300f073c6abe5e985445db5cbb7"
AMENDED_MANIFEST_SHA = (
    "81f007b65b7f09deb779ada330e8da0a8b1cdd0a276c64ae3be9a0e321659f87"
)
ORIGINAL_MANIFEST_SHA = (
    "0004061efdd6d1aec50aa7dc7ab8141a339c6394260fbcdf1681c946a4b382cc"
)
base_path = Path(__file__).with_name("analyze.py")
if hashlib.sha256(base_path.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError("Frozen arithmetic/statistical implementation changed")
spec = importlib.util.spec_from_file_location("native_frozen_analysis", base_path)
assert spec is not None and spec.loader is not None
FROZEN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FROZEN)


def descriptive_interval(values, confidence):
    values = np.asarray(values, dtype=float)
    if len(values) >= 2:
        return FROZEN.interval(values, confidence)
    return dict(
        mean_ms=float(values.mean()) if len(values) else None,
        lower_ms=None,
        upper_ms=None,
        confidence=confidence,
        df=len(values) - 1 if len(values) else None,
        paired_values_ms=values.tolist(),
        interval_unavailable_reason="Fewer than two complete matched blocks",
    )


def terminal_campaign(campaign: dict) -> bool:
    ended = campaign.get("ended_unix")
    if not isinstance(ended, (int, float)) or not math.isfinite(ended):
        return False
    if campaign.get("completed") is True:
        return True
    return (
        campaign.get("completed") is False
        and isinstance(campaign.get("error"), str)
        and bool(campaign["error"])
    )


def analyze(root: Path) -> dict:
    calibration_bytes = (root / "calibration.json").read_bytes()
    calibration_sha = hashlib.sha256(calibration_bytes).hexdigest()
    if (
        calibration_sha != CALIBRATION_SHA
        or calibration_sha != (root / "calibration.sha256").read_text().split()[0]
    ):
        raise ValueError("Frozen calibration changed; no recalibration is permitted")
    calibration = json.loads(calibration_bytes)
    failed = json.loads((root / "block-00-B/status.json").read_text())
    if (
        failed.get("valid") is not False
        or failed.get("error")
        != "TimeoutError('Serving work deadline reached; cleanup reservation begins')"
    ):
        raise ValueError("The original invalid B slot was altered or replaced")
    all_valid = {arm: [] for arm in "ABCD"}
    complete = []
    partial = []
    invalid = []
    for block, order in enumerate(FROZEN.ORDERS):
        rows = {}
        for arm in order:
            path = root / f"block-{block:02d}-{arm}"
            try:
                row = FROZEN.group(path)
                status = json.loads((path / "status.json").read_text())
                source = json.loads((path / "source-attestation.json").read_text())
                expected_work = 465 if block == 0 and arm == "A" else 495
                expected_manifest = (
                    ORIGINAL_MANIFEST_SHA
                    if block == 0 and arm == "A"
                    else AMENDED_MANIFEST_SHA
                )
                if (
                    status.get("work_budget_seconds") != expected_work
                    or status.get("cleanup_reservation_seconds") != 15
                ):
                    raise ValueError(
                        "Group did not use its declared cap/cleanup reserve"
                    )
                if (
                    status["total_seconds"] > expected_work + 15
                    or not source.get("valid")
                    or source.get("manifest_sha256") != expected_manifest
                ):
                    raise ValueError("Group cap or source attestation mismatch")
                row.update(
                    block=block,
                    arm=arm,
                    server_cap_seconds=expected_work + 15,
                    source_manifest_sha256=expected_manifest,
                )
                rows[arm] = row
                all_valid[arm].append(row)
            except Exception as error:
                invalid.append(
                    dict(block=block, arm=arm, path=str(path), error=repr(error))
                )
        if len(rows) == 4:
            complete.append(dict(block=block, rows=rows))
        else:
            partial.append(
                dict(
                    block=block,
                    valid_arms=rows,
                    invalid_arms=[arm for arm in order if arm not in rows],
                )
            )
    if any(row["block"] not in range(1, 8) for row in complete):
        raise ValueError("Only declared complete blocks01–07 are eligible")
    if len({row["request_sha256"] for rows in all_valid.values() for row in rows}) > 1:
        raise ValueError("Treatment request draw changed")
    matched = {arm: [block["rows"][arm] for block in complete] for arm in "ABCD"}
    values = {
        arm: np.asarray([row["mean_ms"] for row in rows])
        for arm, rows in matched.items()
    }
    a, b, c, d = [values[arm] for arm in "ABCD"]
    comparisons = {
        "C-D": descriptive_interval(c - d, 1 - 0.05 / 3),
        "B-D": descriptive_interval(b - d, 1 - 0.05 / 3),
        "(B-A)-(D-C)": descriptive_interval((b - a) - (d - c), 1 - 0.05 / 3),
        "A-D": descriptive_interval(a - d, 0.95),
    }
    for name, control in [("C-D", c), ("B-D", b), ("A-D", a)]:
        control_mean = float(control.mean()) if len(control) else None
        comparisons[name]["percent_of_named_matched_control_mean"] = (
            100 * comparisons[name]["mean_ms"] / control_mean if control_mean else None
        )
    for row in comparisons.values():
        row["n"] = len(complete)
        row["descriptive_only"] = True
        row["numerical_lower_exceeds_frozen_delta"] = (
            row["lower_ms"] > calibration["delta_ms"]
            if row["lower_ms"] is not None
            else None
        )
        row["accepted"] = False
    return {
        "descriptive_only": True,
        "accepted": False,
        "original_eight_valid_block_protocol_complete": False,
        "calibration": calibration,
        "calibration_sha256": calibration_sha,
        "amended_manifest_sha256": AMENDED_MANIFEST_SHA,
        "complete_block_indices": [row["block"] for row in complete],
        "complete_blocks": len(complete),
        "matched_arm_groups": matched,
        "matched_arm_means_ms": {
            arm: float(values[arm].mean()) if len(values[arm]) else None
            for arm in "ABCD"
        },
        "all_valid_arm_groups": all_valid,
        "all_valid_arm_means_ms": {
            arm: float(np.mean([row["mean_ms"] for row in rows])) if rows else None
            for arm, rows in all_valid.items()
        },
        "all_valid_arm_counts": {arm: len(rows) for arm, rows in all_valid.items()},
        "partial_blocks": partial,
        "invalid": invalid,
        "comparisons": comparisons,
        "limits": [
            "Block00B is a retained deadline failure without a primary cadence endpoint; it is never replaced or rescued.",
            "Every comparison and the main four-arm table use complete blocks01–07 only, under the amended cap/harness.",
            "All valid groups and partial block00 are reported separately; no partial quartet contributes to a contrast.",
            "The prospective cap change followed a treatment timeout, so intervals are descriptive and cannot restore original-protocol acceptance.",
            "Task-only numerical qualification, variable acceptance/routed work and potentially different exact-tie KV populations remain limitations.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    campaign = json.loads((args.root / "campaign-cap-amended.json").read_text())
    if not terminal_campaign(campaign):
        raise ValueError("The amended campaign is still live; no early contrasts")
    result = analyze(args.root)
    result["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (args.root / "results-descriptive.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
