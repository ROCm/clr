"""One fresh eight-worker serving group, bounded and fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

_work_deadline = None


def remaining_work_seconds():
    if _work_deadline is None:
        raise RuntimeError("Serving work deadline is not initialized")
    remaining = _work_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Serving work deadline reached; cleanup reservation begins")
    return remaining


def work_deadline_expired(signum, frame):
    raise TimeoutError("Serving work deadline reached; cleanup reservation begins")


def compute_cache_manifest():
    manifests = {}
    roots = [
        Path(os.environ["NATIVE_CACHE"]),
        Path("/sgl-workspace/aiter/aiter/jit"),
    ]
    for root in roots:
        remaining_work_seconds()
        for path in sorted(root.rglob("*")):
            remaining_work_seconds()
            if path.is_file() and path.suffix in (
                ".so",
                ".hsaco",
                ".cubin",
                ".bc",
                ".ptx",
            ):
                digest = hashlib.sha256()
                with path.open("rb") as file:
                    while chunk := file.read(8 * 1024 * 1024):
                        remaining_work_seconds()
                        digest.update(chunk)
                manifests[str(path)] = dict(
                    size=path.stat().st_size, sha256=digest.hexdigest()
                )
    return manifests


def fetch(route, payload=None, timeout=240):
    request = urllib.request.Request(
        "http://127.0.0.1:30000/" + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=min(timeout, remaining_work_seconds())
    ) as response:
        return json.load(response)


def cadence(info):
    found = []

    def walk(value):
        if isinstance(value, dict):
            if "step_time_dict" in value:
                found.append(value["step_time_dict"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(info)
    populated = [item for item in found if item]
    if not populated or any(item != populated[0] for item in populated[1:]):
        raise AssertionError(f"Missing/inconsistent cadence lists: {len(populated)}")
    if set(populated[0]) != {"1"}:
        raise AssertionError(f"Unexpected serving batch census: {populated[0].keys()}")
    return populated[0]["1"]


def validate_correctness(output):
    summary = {}
    for role in ("target", "draft"):
        receipts = [
            json.loads(p.read_text()) for p in output.glob(f"worker-{role}-rank*.json")
        ]
        if sorted(r["rank"] for r in receipts) != list(range(8)):
            raise AssertionError(f"Incomplete {role} rank attestation")
    for rank in range(8):
        records = [
            json.loads(line)
            for line in (output / f"correctness-rank{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        if any(record["failed"] for record in records):
            raise AssertionError(f"Diagnostic mismatch on rank {rank}")
        if not any(
            record["where"] == "capture_failure_and_recapture" for record in records
        ):
            raise AssertionError(f"No real capture lifecycle check on rank {rank}")
        batches = [
            json.loads(line)
            for line in (output / f"runner-inputs-rank{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        keys = {record["key"] for record in batches}
        if len(keys) != 2 or len(batches) != 4:
            raise AssertionError(
                f"Expected dense/sparse graph checks with two real batches each, rank {rank}: {batches}"
            )
        for key in keys:
            pair = [record for record in batches if record["key"] == key]
            if sorted(record["batch"] for record in pair) != [0, 1]:
                raise AssertionError("Missing changed-input graph reuse case")
            if (
                pair[0]["positions"] == pair[1]["positions"]
                or pair[0]["out_cache_loc"] == pair[1]["out_cache_loc"]
            ):
                raise AssertionError(
                    "Runner reuse did not change positions/cache slots"
                )
        for record in batches:
            if (
                record["synchronized"] != 32
                or record["queued"] != 256
                or len(record["input_ids"]) != 4
                or record["alternate_input_ids"] == record["input_ids"]
                or record["input_pattern"] != "alternating_0_1_every_call"
                or not record["differing_projections"]
                or len(record.get("stale_projection_witnesses", [])) != 16
                or not all(
                    item["rejected"]
                    for item in record.get("stale_projection_witnesses", [])
                )
                or not (output / record["expected_states_file"]).is_file()
            ):
                raise AssertionError("Wrong real-runner stress count/width")
        summary[str(rank)] = dict(
            graph_keys=sorted(keys),
            actual_batches=4,
            synchronized=128,
            queued=1024,
            comparison_names=records[-1]["comparisons"],
        )
    (output / "correctness-verdict.json").write_text(
        json.dumps(
            dict(
                valid=True,
                qualification="projection_and_event_dependencies_with_complete_serving",
                full_model_cross_replay_numerical_equivalence=False,
                envelope_sha256="2271908e86c1afbf0a1a194e6ba7721c3acdc742d578f65188ccab05ab7da772",
                ranks=summary,
            ),
            indent=2,
        )
    )


def main():
    global _work_deadline
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("probe", "correctness", "engagement", "performance"),
        required=True,
    )
    parser.add_argument("--cap-seconds", type=int, default=480)
    args = parser.parse_args()
    root = Path(os.environ["NATIVE_ROOT"])
    output = Path(os.environ["NATIVE_ARM_OUTPUT"])
    start = time.monotonic()
    _work_deadline = start + args.cap_seconds - 15
    status = dict(kind=args.kind, valid=False, started_unix=time.time())
    status["work_budget_seconds"] = args.cap_seconds - 15
    status["cleanup_reservation_seconds"] = 15
    server = None
    try:
        signal.signal(signal.SIGALRM, work_deadline_expired)
        signal.setitimer(signal.ITIMER_REAL, remaining_work_seconds())
        argv = json.loads((root / "phase3/server-argv.json").read_text())
        command = [sys.executable, str(root / "phase3/server_entry.py"), *argv]
        if args.kind == "engagement":
            command = [
                "/opt/rocm/bin/rocprofv3",
                "--kernel-trace",
                "--marker-trace",
                "--selected-regions",
                "--output-format",
                "csv",
                "--output-directory",
                str(output / "rocprof"),
                "--output-file",
                "worker-%pid%",
                "--",
                *command,
            ]
        status["command"] = command
        (output / "command.json").write_text(json.dumps(command, indent=2))
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        while time.monotonic() - start < args.cap_seconds - 15:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {server.returncode}")
            try:
                info = fetch("get_server_info", timeout=5)
                break
            except Exception:
                time.sleep(min(2, remaining_work_seconds()))
        else:
            raise TimeoutError("Server startup exceeded group cap")
        status["startup_seconds"] = time.monotonic() - start
        (output / "server-info-ready.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(dict(event="server_ready", **status)), flush=True)
        if args.kind in ("correctness", "engagement"):
            requests = [
                dict(
                    input_ids=[42 + i % 103 for i in range(length)],
                    sampling_params=dict(
                        temperature=0, max_new_tokens=12, ignore_eos=True
                    ),
                    return_logprob=True,
                )
                for length in (
                    (32, 4096, 65, 6144) if args.kind == "correctness" else (8192,)
                )
            ]
            responses = []
            for i, request in enumerate(requests):
                tick = time.monotonic()
                responses.append(fetch("generate", request))
                print(
                    json.dumps(
                        dict(
                            event="correctness_request",
                            index=i,
                            seconds=time.monotonic() - tick,
                        )
                    ),
                    flush=True,
                )
            for response in responses:
                metadata = response["meta_info"]
                if (
                    len(response["output_ids"]) != 12
                    or metadata["completion_tokens"] != 12
                    or metadata["finish_reason"]["type"] != "length"
                ):
                    raise AssertionError(
                        "Serving response did not complete its twelve-token fixture"
                    )
                if not all(
                    math.isfinite(item[0]) for item in metadata["output_token_logprobs"]
                ):
                    raise AssertionError(
                        "Serving response contained nonfinite token log probabilities"
                    )
            (output / "correctness-requests.json").write_text(json.dumps(requests))
            (output / "correctness-responses.json").write_text(json.dumps(responses))
            from worker_hooks import maps_after_warmup

            maps_after_warmup(output)
            if args.kind == "correctness":
                validate_correctness(output)
        else:
            client = root / "phase3/client_entry.py"
            client_base = [
                sys.executable,
                str(client),
                "--model",
                os.environ["NATIVE_MODEL_PATH"],
                "--backend",
                "vllm",
                "--base-url",
                "http://127.0.0.1:30000",
                "--endpoint",
                "/v1/completions",
                "--dataset-name",
                "random",
                "--random-input-len",
                "8192",
                "--random-output-len",
                "1024",
                "--random-range-ratio",
                "0.8",
                "--max-concurrency",
                "1",
                "--request-rate",
                "inf",
                "--ignore-eos",
                "--num-warmups",
                "0",
                "--seed",
                "20260909",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--use-chat-template",
                "--save-result",
                "--save-detailed",
                "--result-filename",
                "bench.json",
            ]
            for phase, count in (("warmup", 33), ("measure", 83)):
                folder = output / phase
                folder.mkdir()
                if phase == "measure":
                    from worker_hooks import maps_after_warmup

                    maps_after_warmup(output)
                    warmed_compute = compute_cache_manifest()
                    (output / "warmup-compute-manifest.json").write_text(
                        json.dumps(warmed_compute, indent=2)
                    )
                    before = fetch("get_server_info")
                    (output / "server-info-before.json").write_text(json.dumps(before))
                    initial = cadence(before)
                    log_offset = (output / "server.log").stat().st_size
                command = [
                    *client_base,
                    "--num-prompts",
                    str(count),
                    "--result-dir",
                    str(folder),
                ]
                tick = time.monotonic()
                remaining = remaining_work_seconds()
                with (folder / "client.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=remaining,
                        env=dict(
                            os.environ,
                            NATIVE_REQUEST_CAPTURE_PATH=str(folder / "requests.json"),
                        ),
                    )
                elapsed = time.monotonic() - tick
                status[f"{phase}_seconds"] = elapsed
                result = json.loads((folder / "bench.json").read_text())
                if result["completed"] != count or any(result.get("errors", [])):
                    raise AssertionError(
                        f"Incomplete {phase}: {result['completed']}/{count}"
                    )
                if any(length <= 2048 for length in result["input_lens"]):
                    raise AssertionError("Unexpected dense-only measured request")
                print(
                    json.dumps(
                        dict(
                            event=phase, seconds=elapsed, completed=result["completed"]
                        )
                    ),
                    flush=True,
                )
                if phase == "measure":
                    after = fetch("get_server_info")
                    (output / "server-info-after.json").write_text(json.dumps(after))
                    final = cadence(after)
                    if (
                        final[: len(initial)] != initial
                        or len(final) <= len(initial) + 1
                    ):
                        raise AssertionError(
                            "Cadence counter reset or no completed intervals"
                        )
                    raw = final[len(initial) :]
                    (output / "cadence.json").write_text(
                        json.dumps(
                            dict(
                                before_index=len(initial),
                                after_index=len(final),
                                boundary_only=raw[0],
                                samples_seconds=raw[1:],
                                all_window_seconds=raw,
                            )
                        )
                    )
                    import numpy as np

                    samples = np.asarray(raw[1:]) * 1000
                    status["cadence_ms"] = dict(
                        count=len(samples),
                        mean=float(samples.mean()),
                        median=float(np.median(samples)),
                        p90=float(np.percentile(samples, 90)),
                        p95=float(np.percentile(samples, 95)),
                        p99=float(np.percentile(samples, 99)),
                        maximum=float(samples.max()),
                        pauses_ge500_ms=samples[samples >= 500].tolist(),
                    )
                    status["client_duration"] = result["duration"]
                    if result["duration"] < 60:
                        raise AssertionError(
                            "Measured serving below frozen 60-second minimum"
                        )
                    with (output / "server.log").open("rb") as file:
                        file.seek(log_offset)
                        measured_log = file.read().decode(errors="replace")
                    accept_lengths = [
                        float(value)
                        for value in re.findall(r"accept len: ([0-9.]+)", measured_log)
                    ]
                    decode_lines = [
                        line
                        for line in measured_log.splitlines()
                        if "accept len:" in line
                    ]
                    if any("cuda graph: True" not in line for line in decode_lines):
                        raise AssertionError(
                            "Measured target serving iteration did not use its captured graph"
                        )
                    if len(accept_lengths) != len(raw):
                        raise AssertionError(
                            f"Cadence/acceptance log count mismatch: {len(raw)} versus {len(accept_lengths)}"
                        )
                    (output / "acceptance.json").write_text(
                        json.dumps(
                            dict(
                                all_window=accept_lengths,
                                boundary_only=accept_lengths[0],
                                primary=accept_lengths[1:],
                            )
                        )
                    )
                    from collections import Counter

                    status["acceptance_histogram"] = dict(Counter(accept_lengths[1:]))
                    status["mean_accept_length"] = float(np.mean(accept_lengths[1:]))
                    final_compute = compute_cache_manifest()
                    (output / "final-compute-manifest.json").write_text(
                        json.dumps(final_compute, indent=2)
                    )
                    if final_compute != warmed_compute:
                        raise AssertionError(
                            "Compute artifact set/bytes changed in measured window"
                        )
                    if any(
                        marker in measured_log
                        for marker in (
                            "JIT build",
                            "Start compiling",
                            "Compiling kernels",
                            "HSA_STATUS_ERROR_EXCEPTION",
                            "Traceback",
                        )
                    ):
                        raise AssertionError(
                            "Compilation or runtime error in measured window"
                        )
        status["valid"] = True
    except BaseException as error:
        status["error"] = repr(error)
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _work_deadline = None
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                remaining = args.cap_seconds - (time.monotonic() - start)
                server.wait(timeout=max(0, min(20, remaining - 5)))
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                try:
                    server.wait(
                        timeout=max(
                            0.1, min(5, args.cap_seconds - (time.monotonic() - start))
                        )
                    )
                except subprocess.TimeoutExpired:
                    status["valid"] = False
                    status["server_cleanup_timeout"] = True
        cleanup = []
        worker_pids = sorted(
            {
                json.loads(path.read_text())["pid"]
                for path in output.glob("worker-*-rank*.json")
            }
        )
        for pid in worker_pids:
            proc = Path(f"/proc/{pid}")
            if proc.exists():
                try:
                    environment = (proc / "environ").read_bytes().split(b"\0")
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except PermissionError as error:
                    cleanup.append(dict(pid=pid, unverified_survivor=repr(error)))
                    status["valid"] = False
                    continue
                tag = f"NATIVE_ARM_OUTPUT={output}".encode()
                if tag in environment and proc.stat().st_uid == os.getuid():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        continue
                    cleanup.append(dict(pid=pid, killed_owned_survivor=True))
                else:
                    cleanup.append(dict(pid=pid, unverified_survivor=True))
                    status["valid"] = False
        cleanup_deadline = min(start + args.cap_seconds, time.monotonic() + 5)
        remaining_pids = [pid for pid in worker_pids if Path(f"/proc/{pid}").exists()]
        while remaining_pids and time.monotonic() < cleanup_deadline:
            time.sleep(0.1)
            remaining_pids = [
                pid for pid in remaining_pids if Path(f"/proc/{pid}").exists()
            ]
        if remaining_pids:
            status["valid"] = False
            status["worker_cleanup_survivors"] = remaining_pids
        status["cleanup"] = cleanup
        status["total_seconds"] = time.monotonic() - start
        if status["total_seconds"] > args.cap_seconds:
            status["valid"] = False
            status["cap_exceeded"] = True
        status["ended_unix"] = time.time()
        (output / "status.json").write_text(json.dumps(status, indent=2))
        print("NATIVE_GROUP_DONE " + json.dumps(status), flush=True)
    return 0 if status["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
