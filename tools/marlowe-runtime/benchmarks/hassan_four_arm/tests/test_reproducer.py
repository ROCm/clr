"""CPU-only checks for ownership, receipts, boundaries and archived arithmetic."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import analyze
import reproduce
import worker

REPO = Path(__file__).resolve().parents[1]


def put(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ReproducerTests(unittest.TestCase):
    def test_fixed_prospective_population_has_no_historical_hole(self):
        slots = reproduce.slots()
        self.assertEqual(len(slots), 40)
        self.assertEqual(len(set(slots)), 40)
        self.assertEqual(slots[:8], [(f"aa-{i:02d}-A", "A") for i in range(8)])
        self.assertEqual(slots[8:12], [(f"block-00-{a}", a) for a in "ABDC"])
        self.assertEqual(slots[-4:], [(f"block-07-{a}", a) for a in "DACB"])
        self.assertEqual(
            reproduce.CONTRACT["prospective_protocol"]["server_cap_seconds"], 510
        )

    def test_archived_arithmetic_and_historical_invalidity(self):
        result = analyze.archived(None)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["complete_block_indices"], list(range(1, 8)))
        self.assertEqual(
            [(x["block"], x["arm"]) for x in result["invalid"]], [(0, "B")]
        )
        self.assertLess(result["comparisons"]["C-D"]["upper_ms"], 0)

    def test_wrong_numeric_result_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze.compare_numbers({"mean_ms": 3.5}, {"mean_ms": 3.6})
        with self.assertRaises(ValueError):
            analyze.compare_numbers(float("nan"), 1.0)

    def test_retained_and_relocated_payload_hashes(self):
        retained = json.loads((REPO / "contracts/retained-harness.json").read_text())
        relocated = {
            x["file"]: x
            for x in json.loads((REPO / "contracts/relocations.json").read_text())
        }
        for name, expected in retained.items():
            if name in relocated:
                self.assertEqual(relocated[name]["original_sha256"], expected)
                expected = relocated[name]["portable_sha256"]
            self.assertEqual(reproduce.sha(REPO / name), expected, name)

    def test_atomic_claim_across_output_roots_and_loser_cannot_release(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "90001"}),
        ):
            parent = Path(tmp)
            roots = [parent / "first", parent / "second"]
            for i, root in enumerate(roots):
                put(
                    root / "claim-request.json",
                    dict(job_id="90001", uid=os.getuid(), root=str(root), token=str(i)),
                )
            with mock.patch.object(worker, "cache_path", return_value=parent / "cache"):

                def acquire(root):
                    try:
                        worker.claim(root)
                        return True
                    except FileExistsError:
                        return False

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    wins = list(pool.map(acquire, roots))
                self.assertEqual(sum(wins), 1)
                worker.require_claim(roots[wins.index(True)])
                with self.assertRaises((ValueError, FileNotFoundError)):
                    worker.require_claim(roots[wins.index(False)])

    def test_failed_claim_never_enters_scancel_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = {"output_root": str(root)}
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(reproduce, "owned_job", return_value={}),
                mock.patch.object(reproduce, "stage", return_value=root),
                mock.patch.object(reproduce, "step_prefix", return_value=["srun"]),
                mock.patch.object(
                    reproduce,
                    "command",
                    side_effect=subprocess.CalledProcessError(1, ["claim"]),
                ),
                mock.patch.object(reproduce.subprocess, "run") as run,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    reproduce.run(site, 90001)
                run.assert_not_called()

    def test_partial_preparation_cleans_only_its_claimed_cache(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "90001"}),
        ):
            parent = Path(tmp)
            root, cache = parent / "run", parent / "cache"
            put(
                root / "claim-request.json",
                dict(job_id="90001", uid=os.getuid(), root=str(root), token="one"),
            )
            put(
                root / "phase4/campaign.json",
                dict(
                    groups=[], invalid_attempts=[], ended_unix=1, error="prepare failed"
                ),
            )
            untouched = parent / "unrelated"
            untouched.write_text("keep")
            with mock.patch.object(worker, "cache_path", return_value=cache):
                worker.claim(root)
                cache.mkdir()
                put(
                    cache / "owner.json",
                    dict(job_id="90001", uid=os.getuid(), root=str(root)),
                )
                (cache / "seed").mkdir()
                (cache / "seed/partial").write_text("partial")
                worker.cleanup(root)
            self.assertFalse(cache.exists())
            self.assertEqual(untouched.read_text(), "keep")
            self.assertTrue(
                json.loads((root / "cache-cleanup.json").read_text())["absent"]
            )

    def _cleanup_campaign(
        self,
        root,
        *,
        cache_rc=0,
        recheck_timeout=False,
        health_timeout=False,
        visibility_timeout=False,
        admission_left=None,
    ):
        (root / "phase4").mkdir()
        cancelled = []

        def command(argv, **_kwargs):
            if "claim" in argv:
                request = json.loads((root / "claim-request.json").read_text())
                if not visibility_timeout:
                    put(root / "claim.json", request)
                return subprocess.CompletedProcess(
                    argv, 0, "NATIVE_CLAIM_DONE " + json.dumps(request) + "\n", ""
                )
            if "authorize-release" in argv and recheck_timeout:
                raise subprocess.TimeoutExpired(argv, 60)
            return subprocess.CompletedProcess(argv, 0, "", "")

        def process(argv, **_kwargs):
            self.assertNotIn("group", argv)
            if "health" in argv and health_timeout:
                raise subprocess.TimeoutExpired(argv, 60)
            if "cleanup" in argv:
                # Even a true-looking stale receipt must not rescue a failed step.
                put(root / "cache-cleanup.json", {"absent": True})
                return subprocess.CompletedProcess(argv, cache_rc, "", "")
            if argv[0] == "scancel":
                cancelled.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                reproduce, "owned_job", return_value={"seconds_left": admission_left}
            ),
            mock.patch.object(reproduce, "stage", return_value=root),
            mock.patch.object(reproduce, "step_prefix", return_value=["srun"]),
            mock.patch.object(
                reproduce,
                "slots",
                return_value=[] if admission_left is None else [("aa-00-A", "A")],
            ),
            mock.patch.object(reproduce, "command", side_effect=command),
            mock.patch.object(reproduce.subprocess, "run", side_effect=process),
            mock.patch.object(
                reproduce,
                "verify_claim_visibility",
                side_effect=ValueError("Claim receipt visibility timeout")
                if visibility_timeout
                else None,
            ),
        ):
            if (
                cache_rc
                or health_timeout
                or visibility_timeout
                or admission_left is not None
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Claim receipt visibility timeout"
                    if visibility_timeout
                    else "Insufficient allocation time"
                    if admission_left is not None
                    else "cleanup was not fully verified",
                ):
                    reproduce.run({"output_root": str(root)}, 90001)
            else:
                reproduce.run({"output_root": str(root)}, 90001)
        self.assertEqual(cancelled, [["scancel", "90001"]])
        receipt = json.loads((root / "release.json").read_text())
        self.assertTrue(receipt["released_verified"])
        return receipt

    def test_nonzero_cache_cleanup_prevents_success_even_with_absent_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = self._cleanup_campaign(Path(tmp), cache_rc=1)
        self.assertFalse(receipt["cache_absent_verified"])
        self.assertEqual(receipt["cache_cleanup_returncode"], 1)

    def test_failed_node_recheck_still_releases_only_initially_claimed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = self._cleanup_campaign(Path(tmp), recheck_timeout=True)
        self.assertIn("TimeoutExpired", receipt["claim_recheck_error"])
        self.assertEqual(receipt["release_authority"], "verified initial atomic claim")

    def test_health_timeout_still_attempts_cache_cleanup_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = self._cleanup_campaign(Path(tmp), health_timeout=True)
        self.assertFalse(receipt["health_verified"])
        self.assertEqual(receipt["cache_cleanup_returncode"], 0)

    def test_successful_claim_with_missing_disk_receipt_still_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = self._cleanup_campaign(Path(tmp), visibility_timeout=True)
        self.assertTrue(receipt["cache_absent_verified"])

    def test_claim_visibility_is_bounded_and_refuses_parsed_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.object(reproduce.time, "monotonic", side_effect=[0, 59, 60]),
                mock.patch.object(reproduce.time, "sleep"),
            ):
                with self.assertRaisesRegex(ValueError, "visibility timeout"):
                    reproduce.verify_claim_visibility(root, {"owner": "one"})
            put(root / "claim.json", {"owner": "another"})
            with self.assertRaisesRegex(ValueError, "claim receipt mismatch"):
                reproduce.verify_claim_visibility(root, {"owner": "one"})

    def test_admission_reserves_complete_terminal_cleanup_before_new_group(self):
        guard = reproduce.CONTRACT["prospective_protocol"][
            "allocation_admission_seconds"
        ]
        self.assertEqual(guard, 1170)
        self.assertGreaterEqual(guard, 690 + 60 + 60 + 90 + 60 + 15 + 122 + 15)
        with tempfile.TemporaryDirectory() as tmp:
            receipt = self._cleanup_campaign(Path(tmp), admission_left=guard - 1)
        self.assertTrue(receipt["released_verified"])

    def test_release_layout_verifies_installed_launch_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib").mkdir()
            (root / "lib/library.so.1").write_bytes(b"library")
            (root / "lib/library.so").symlink_to("library.so.1")
            for name in ("run", "verify.py"):
                (root / name).write_text(name)
            runtime = {
                "libraries": {"library.so.1": reproduce.sha(root / "lib/library.so.1")},
                "aliases": {"library.so": "library.so.1"},
                "recipe_sha256": {
                    "build.sh": "build input is not installed at release root",
                    **{
                        name: reproduce.sha(root / name)
                        for name in ("run", "verify.py")
                    },
                },
            }
            put(root / "manifest.json", runtime)
            worker.verify_runtime_release(root, runtime)
            (root / "run").write_text("changed")
            with self.assertRaisesRegex(ValueError, "launch file"):
                worker.verify_runtime_release(root, runtime)
            put(root / "manifest.json", {**runtime, "unqualified": True})
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                worker.verify_runtime_release(root, runtime)

    def test_archived_analysis_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "analyze.py"),
                    "archived",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(result.stdout)["complete_block_indices"], list(range(1, 8))
            )
            self.assertFalse(json.loads(output.read_text())["accepted"])

    def test_live_prospective_campaign_with_invalid_history_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(
                root / "phase4/campaign.json",
                {"groups": [], "invalid_attempts": [{"error": "old"}]},
            )
            with self.assertRaisesRegex(ValueError, "not terminal"):
                analyze.prospective(root)

    def test_cleanup_refuses_foreign_owner_and_live_worker_evidence(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "90001"}),
        ):
            root, cache = Path(tmp) / "run", Path(tmp) / "cache"
            put(
                root / "claim-request.json",
                dict(job_id="90001", uid=os.getuid(), root=str(root), token="one"),
            )
            put(root / "phase4/campaign.json", dict(groups=[{}], invalid_attempts=[]))
            with mock.patch.object(worker, "cache_path", return_value=cache):
                worker.claim(root)
                cache.mkdir()
                with self.assertRaises(ValueError):
                    worker.cleanup(root)
                self.assertTrue(cache.exists())

    def _workers(self, output: Path):
        dispatch = json.loads((REPO / "contracts/projection-dispatch.json").read_text())
        for role in ("target", "draft"):
            for rank in range(8):
                row = dict(
                    role=role,
                    rank=rank,
                    pid=1000 + rank,
                    env=dict(
                        SLURM_JOB_ID="90001",
                        SLURM_STEP_ID="3",
                        SLURM_GPUS_ON_NODE="8",
                        NATIVE_CORRECTNESS="0",
                        NATIVE_PROFILE="0",
                        **reproduce.CONTRACT["arms"]["A"],
                    ),
                    runtime=dict(gpu_count=8),
                    layers=list(range(10 if role == "target" else 1)),
                    weight_bytes=5947301024,
                    invoked_indexers=[0, 1, 2, 6],
                    selected_projection_dispatch=dispatch,
                    decode_runner_class="EagerRunner"
                    if role == "draft"
                    else "DecodeCudaGraphRunner",
                )
                put(output / f"worker-{role}-rank{rank}.json", row)

    def test_worker_census_rejects_stale_duplicate_and_missing_records(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"SLURM_JOB_ID": "90001"}),
        ):
            output = Path(tmp)
            self._workers(output)
            self.assertEqual(
                worker.validate_workers(output, "A", "3"), list(range(1000, 1008))
            )
            with self.assertRaises(ValueError):
                worker.validate_workers(output, "A", "4")
            path = output / "worker-target-rank7.json"
            saved = json.loads(path.read_text())
            changed = dict(saved, pid=1000)
            put(path, changed)
            with self.assertRaises(ValueError):
                worker.validate_workers(output, "A", "3")
            put(path, dict(saved, rank=6))
            with self.assertRaises(ValueError):
                worker.validate_workers(output, "A", "3")
            path.unlink()
            with self.assertRaises(ValueError):
                worker.validate_workers(output, "A", "3")

    def test_real_serving_work_deadline_interrupts_blocking_operation(self):
        spec = importlib.util.spec_from_file_location(
            "timed_runner", REPO / "harness/run_group.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        previous = signal.signal(signal.SIGALRM, runner.work_deadline_expired)
        try:
            runner._work_deadline = time.monotonic() + 0.02
            signal.setitimer(signal.ITIMER_REAL, runner.remaining_work_seconds())
            with self.assertRaises(TimeoutError):
                time.sleep(0.2)
            with self.assertRaises(TimeoutError):
                runner.remaining_work_seconds()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def test_source_patch_applies_to_exact_base_when_available(self):
        base = os.environ.get("SGLANG_BASE_REPO")
        if not base:
            self.skipTest("Set SGLANG_BASE_REPO for the exact-object patch smoke")
        tracked = [
            "python/sglang/srt/environ.py",
            "python/sglang/srt/layers/attention/dsa/dsa_indexer.py",
            "python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py",
        ]
        archive = subprocess.check_output(
            ["git", "-C", base, "archive", reproduce.CONTRACT["sglang_base"], *tracked]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(root, filter="data")
            subprocess.run(
                [
                    "git",
                    "apply",
                    "--check",
                    str(REPO / "patches/sglang-indexer-events.patch"),
                ],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "apply", str(REPO / "patches/sglang-indexer-events.patch")],
                cwd=root,
                check=True,
            )
            for name, expected in reproduce.CONTRACT["qualified_files"].items():
                self.assertEqual(reproduce.sha(root / name), expected)


if __name__ == "__main__":
    unittest.main()
