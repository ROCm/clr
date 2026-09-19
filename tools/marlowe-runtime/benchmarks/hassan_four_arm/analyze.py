"""Read-only archived or prospective reduction using the retained statistics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def harness_root() -> Path:
    return REPO / ("harness" if (REPO / "harness").exists() else "phase3")


def verified_harness(name: str) -> Path:
    path = harness_root() / name
    retained = json.loads((REPO / "contracts/retained-harness.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != retained["harness/" + name]:
        raise ValueError("Retained analyzer source changed: " + name)
    return path


FROZEN = load_module("frozen_arithmetic", verified_harness("analyze.py"))


def portable(value):
    if isinstance(value, dict):
        return {key: portable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [portable(child) for child in value]
    if isinstance(value, str):
        return re.sub(r"/[^\s'\"]+/(aa-\d\d-A|block-\d\d-[ABCD])", r"raw/\1", value)
    return value


def compare_numbers(actual, expected, path="root") -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Field mismatch: {path}")
        for key in expected:
            compare_numbers(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Length mismatch: {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_numbers(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"Numerical mismatch: {path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Value mismatch: {path}")


def archived(raw: Path | None) -> dict:
    expected_path = REPO / "expected/results-descriptive.json"
    provenance = json.loads((REPO / "expected/provenance.json").read_text())
    if (
        hashlib.sha256(expected_path.read_bytes()).hexdigest()
        != provenance["portable_result_sha256"]
    ):
        raise ValueError("Archived result bytes changed")
    expected = json.loads(expected_path.read_text())
    if raw is not None:
        module = load_module(
            "archived_contract", verified_harness("analyze_archived.py")
        )
        campaign = json.loads((raw / "campaign-cap-amended.json").read_text())
        if not module.terminal_campaign(campaign):
            raise ValueError("Archived input campaign is not terminal")
        result = module.analyze(raw)
        result["analysis_source_sha256"] = hashlib.sha256(
            (harness_root() / "analyze_archived.py").read_bytes()
        ).hexdigest()
        compare_numbers(portable(result), expected)
    else:
        # This compact check recomputes published group-level statistics. Raw
        # interval/receipt verification requires the optional full evidence.
        result = expected
        values = {
            arm: np.asarray(
                [row["mean_ms"] for row in result["matched_arm_groups"][arm]]
            )
            for arm in "ABCD"
        }
        for arm in "ABCD":
            compare_numbers(
                float(values[arm].mean()), result["matched_arm_means_ms"][arm]
            )
        a, b, c, d = (values[arm] for arm in "ABCD")
        comparisons = {
            "C-D": c - d,
            "B-D": b - d,
            "(B-A)-(D-C)": (b - a) - (d - c),
            "A-D": a - d,
        }
        for name, values in comparisons.items():
            stored = result["comparisons"][name]
            actual = FROZEN.interval(values, stored["confidence"])
            compare_numbers(actual, {key: stored[key] for key in actual})
        if (
            result["accepted"]
            or result["complete_block_indices"] != list(range(1, 8))
            or result["all_valid_arm_counts"] != dict(A=8, B=7, C=8, D=8)
        ):
            raise ValueError("Archived validity/population changed")
    return result


def prospective(root: Path) -> dict:
    campaign = json.loads((root / "phase4/campaign.json").read_text())
    ended = campaign.get("ended_unix")
    if (
        not isinstance(ended, (int, float))
        or not math.isfinite(ended)
        or not (campaign.get("completed") is True or campaign.get("error"))
    ):
        raise ValueError("Prospective campaign is not terminal")
    if campaign.get("protocol") != "prospective_uniform_510_seconds":
        raise ValueError(
            "Wrong protocol; historical cap amendments are not prospective runs"
        )
    records = {row["name"]: row for row in campaign["groups"]}
    if len(records) != len(campaign["groups"]):
        raise ValueError("Duplicate admitted groups")
    known_slots = [f"aa-{i:02d}-A" for i in range(8)] + [
        f"block-{block:02d}-{arm}"
        for block, order in enumerate(FROZEN.ORDERS)
        for arm in order
    ]
    if [row["name"] for row in campaign["groups"]] != known_slots[: len(records)]:
        raise ValueError("Admitted groups are not the fixed prefix")
    manifest_path = root / "execution-manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())

    def group(name: str) -> dict:
        if name not in records:
            raise ValueError("Missing or invalid scheduled slot")
        path = root / "phase4" / name
        status = json.loads((path / "status.json").read_text())
        if (
            status != records[name]["status"]
            or records[name]["step_rc"] != 0
            or status.get("cleanup") != []
        ):
            raise ValueError("Campaign/group/cleanup mismatch")
        if (
            status.get("work_budget_seconds") != 495
            or status.get("cleanup_reservation_seconds") != 15
            or status["total_seconds"] > 510
        ):
            raise ValueError("Wrong prospective cap")
        source = json.loads((path / "source-attestation.json").read_text())
        if (
            not source["valid"]
            or source["manifest_sha256"] != manifest_sha
            or source["verified_files"] != manifest["execution_files"]
        ):
            raise ValueError("Source binding mismatch")
        return FROZEN.group(path)

    aa = [group(f"aa-{i:02d}-A") for i in range(8)]
    calibration_path = root / "phase4/calibration.json"
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != campaign.get(
        "calibration_sha256"
    ):
        raise ValueError("Calibration was not frozen by this campaign")
    calibration = json.loads(calibration_path.read_text())
    compare_numbers(aa, calibration["groups"])
    values = [row["mean_ms"] for row in aa]
    delta = max(
        0.005,
        0.005 * float(np.mean(values)),
        2 * max(abs(values[i + 1] - values[i]) for i in range(0, 8, 2)),
    )
    compare_numbers(delta, calibration["delta_ms"])
    complete, partial, invalid = [], [], []
    all_valid = {arm: [] for arm in "ABCD"}
    for block, order in enumerate(FROZEN.ORDERS):
        rows = {}
        for arm in order:
            name = f"block-{block:02d}-{arm}"
            try:
                rows[arm] = group(name)
                all_valid[arm].append(rows[arm])
            except Exception as error:
                invalid.append(dict(block=block, arm=arm, error=repr(error)))
        (complete if len(rows) == 4 else partial).append(dict(block=block, rows=rows))
    draws = {
        row["request_sha256"]
        for row in aa + [row for rows in all_valid.values() for row in rows]
    }
    if len(draws) != 1:
        raise ValueError("Request draw changed")
    matched = {
        arm: np.asarray([block["rows"][arm]["mean_ms"] for block in complete])
        for arm in "ABCD"
    }
    a, b, c, d = (matched[arm] for arm in "ABCD")
    comparisons = {}
    for name, values in {
        "C-D": c - d,
        "B-D": b - d,
        "(B-A)-(D-C)": (b - a) - (d - c),
        "A-D": a - d,
    }.items():
        confidence = 0.95 if name == "A-D" else 1 - 0.05 / 3
        row = (
            FROZEN.interval(values, confidence)
            if len(values) >= 2
            else dict(
                mean_ms=float(values.mean()) if len(values) else None,
                lower_ms=None,
                upper_ms=None,
                confidence=confidence,
                df=len(values) - 1 if len(values) else None,
                paired_values_ms=values.tolist(),
            )
        )
        row.update(n=len(complete), descriptive_only=True, accepted=False)
        if name in ("C-D", "B-D", "A-D") and len(values):
            row["percent_of_named_matched_control_mean"] = (
                100 * row["mean_ms"] / float(matched[name[0]].mean())
            )
        comparisons[name] = row
    return dict(
        protocol=campaign["protocol"],
        accepted=False,
        descriptive_only=True,
        calibration=calibration,
        complete_block_indices=[row["block"] for row in complete],
        matched_arm_means_ms={
            arm: float(values.mean()) if len(values) else None
            for arm, values in matched.items()
        },
        all_valid_arm_counts={arm: len(rows) for arm, rows in all_valid.items()},
        all_valid_arm_groups=all_valid,
        partial_blocks=partial,
        invalid=invalid,
        comparisons=comparisons,
        limits=[
            "Portable launcher has CPU smoke coverage but was not GPU-rerun during packaging.",
            "Task-only Q/K/selector qualification; no full-model equivalence or production acceptance.",
            "Paired-t intervals assume independent approximately normal block differences; small n and work variability remain limits.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("archived", "prospective"))
    parser.add_argument(
        "--raw", type=Path, help="Optional full historical evidence root"
    )
    parser.add_argument("--run", type=Path, help="Prospective output root")
    parser.add_argument(
        "--output", type=Path, help="New output file; never overwrites raw evidence"
    )
    args = parser.parse_args()
    if args.mode == "prospective" and args.run is None:
        parser.error("prospective requires --run")
    result = archived(args.raw) if args.mode == "archived" else prospective(args.run)
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "matched_arm_means_ms",
                    "complete_block_indices",
                    "accepted",
                    "descriptive_only",
                )
            },
            indent=2,
        )
    )
