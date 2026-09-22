#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and run bounded graph-frontier regressions or standalone microbenchmarks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

if not __debug__:
    raise RuntimeError("Run without -O: assertions validate the test results")

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "burst": "graph_signal_generations.cpp",
    "boundary": "graph_frontier_boundaries.cpp",
    "idle": "graph_frontier_idle.cpp",
    "hostfed": "graph_frontier_host_fed.cpp",
    "alias": "graph_boundary_alias.cpp",
    "lane": "graph_lane_lifetime.cpp",
    "pool": "../native_event_wait/native_pool_pressure.cpp",
    "graph_microbenchmark": "graph_microbenchmark.cpp",
    "waiter": "../native_event_wait/minimal_wait.cpp",
}
LEGACY_CONTROLS = (
    "GPU_NATIVE_EVENT_WAIT", "GPU_GRAPH_NODE_COUNT_PLACEMENT",
    "GPU_GRAPH_DIAGNOSTIC_SPARE", "GPU_GRAPH_DIAGNOSTIC_QUALIFIED_SPARE",
    "GPU_GRAPH_DIAGNOSTIC_LANE_RETIRE", "GPU_GRAPH_DIAGNOSTIC_COVERED_TAIL",
    "GPU_GRAPH_DIAGNOSTIC_FUSE_DEPS", "GPU_GRAPH_DIAGNOSTIC_FUSE_FINAL",
    "GPU_GRAPH_DIAGNOSTIC_KERNEL_RETIRE", "GPU_GRAPH_DIAGNOSTIC_LOCAL_SIGNALS",
    "GPU_GRAPH_DIAGNOSTIC_FRONTIER", "GPU_GRAPH_DIAGNOSTIC_FRONTIER_DISTRIBUTED",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def clean_environment():
    prefixes = ("GPU_", "DEBUG_HIP_", "DEBUG_CLR_", "AMD_", "ROC_AQL_", "HSA_")
    return {k: v for k, v in os.environ.items()
            if not k.startswith(prefixes) and k not in
            ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_DEBUG")}


def cases(benchmarks):
    if benchmarks:
        return [{"kind": "graph_microbenchmark"}, {"kind": "waiter"}]
    result = [{"kind": kind, "capacity": capacity} for kind, capacity in (
        ("burst", 2), ("burst", 8), ("burst", 32), ("boundary", 8),
        ("boundary", 1024), ("idle", 8), ("hostfed", 2), ("alias", 1024))]
    result.extend([
        {"kind": "boundary", "capacity": 8, "legacy_zeroed": True},
        {"kind": "lane", "queues": 1},
        {"kind": "lane", "queues": 4},
        {"kind": "lane", "queues": 4, "fault": 3},
        {"kind": "pool"},
    ])
    return result


def inspect_frontiers(log):
    active = {}
    launches = chained = peak = 0
    for line in re.findall(r"GRAPH_FRONTIER_[A-Z_]+ [^\r\n]*", log):
        event = line.split()[0]
        values = dict(re.findall(r"(\w+)=(\S+)", line))
        key = (values.get("graph"), values.get("serial"))
        if event == "GRAPH_FRONTIER_BEGIN":
            assert key not in active
            assert values["generation"] not in [v["generation"] for v in active.values()]
            active[key] = dict(values, ended=False)
            launches += 1
            chained += int(values["chained"])
            peak = max(peak, len(active))
        elif event == "GRAPH_FRONTIER_END":
            assert key in active and not active[key]["ended"]
            active[key]["ended"] = True
        elif event == "GRAPH_FRONTIER_RELEASE":
            assert key in active and active[key]["ended"]
            assert active[key]["generation"] == values["generation"]
            assert values["recycled"] == "1"
            del active[key]
    assert not active, "Unretired or reused private generation"
    return {"launches": launches, "chained": chained, "peak": peak}


def inspect_result(case, stdout, stderr, benchmarks):
    kind = case["kind"]
    if benchmarks:
        assert "GRAPH_FRONTIER_" not in stderr, "Tracing must be disabled in timing runs"
        rows = list(csv.DictReader(stdout.splitlines()))
        assert len(rows) == (96 if kind == "graph_microbenchmark" else 120)
        assert all(row["correct"] == "1" and float(row["gpu_us"]) > 0 for row in rows)
        if kind == "waiter":
            assert all(row["pending"] == "1" for row in rows if row["case"] == "pending_wait")
        return {"timing_rows": len(rows)}
    if kind == "pool":
        assert "watchdog=0" in stdout and "pending=2300" in stdout and "actual=2300" in stdout
        return {"enqueue_progress": True}
    frontier = inspect_frontiers(stderr)
    if kind == "lane":
        rows = list(csv.DictReader(stdout.splitlines()))
        assert len(rows) == (1 if case.get("fault") else 6)
        assert all(row["correct"] == "1" for row in rows)
        assert (frontier["launches"] > 0) == (case.get("queues", 4) > 1)
        return {"rows": len(rows), "frontier": frontier}
    result = json.loads(stdout)
    assert result["passed"]
    capacity = case.get("capacity", 1024)
    assert frontier["launches"] > 0 and frontier["peak"] <= capacity
    fallbacks = [dict(re.findall(r"(\w+)=(\S+)", line)) for line in
                 re.findall(r"GRAPH_FRONTIER_CAPACITY [^\r\n]*", stderr)]
    if kind != "idle" and capacity < 64:
        assert fallbacks, "Capacity fallback was not exercised"
    assert all(int(v["size"]) == int(v["limit"]) == capacity and
               v["action"] == "fallback" for v in fallbacks)
    if kind == "burst":
        assert result["checked_values"] == 448 and result["updates"] == 448
    elif kind == "boundary":
        assert result["checked_values"] == 2210 and result["callback_count"] == 1
    elif kind == "alias":
        assert result["checked_values"] == 130 and frontier["launches"] == 64
        assert frontier["chained"] >= 62
    elif kind == "hostfed":
        assert result["checked_values"] == 16 and not result["watchdog_rescued"]
        assert result["submit_ms"] < 1000
    elif kind == "idle":
        assert result["checked_values"] == 2 and result["destroy_ms"] < 100
        assert (stderr.index("FRONTIER_IDLE phase=before_destroy") <
                stderr.index("GRAPH_FRONTIER_RELEASE ") <
                stderr.index("FRONTIER_IDLE phase=idle_end"))
    return {"frontier": frontier, "capacity_fallbacks": len(fallbacks), "result": result}


def mapped_libraries(log, libraries):
    mapped = {}
    for name, expected in libraries.items():
        paths = {str(Path(line.split("calling init:", 1)[1].strip()).resolve())
                 for line in log.splitlines() if "calling init:" in line and name in line}
        assert paths == {expected["path"]}, (name, paths, expected)
        mapped[name] = expected
    return mapped


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["runner_sha256"] == digest(__file__)
    assert [run["case"] for run in manifest["runs"]] == cases(manifest["benchmarks"])
    for source, expected in manifest["sources"].items():
        assert digest(ROOT / source) == expected
    for name, expected in manifest["binaries"].items():
        assert digest(output / "bin" / name) == expected
    for run in manifest["runs"]:
        assert run["returncode"] == 0
        for name, expected in run["files"].items():
            assert digest(output / name) == expected
        stdout = (output / (run["name"] + ".stdout")).read_text()
        stderr = (output / (run["name"] + ".stderr")).read_text()
        assert run["mapped"] == mapped_libraries(stderr, manifest["libraries"])
        assert run["observed"] == inspect_result(run["case"], stdout, stderr, manifest["benchmarks"])
    print(json.dumps({"passed": True, "processes": len(manifest["runs"]),
                      "benchmarks": manifest["benchmarks"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    parser.add_argument("--gpu-arch", default="gfx950")
    parser.add_argument("--benchmarks-only", action="store_true")
    parser.add_argument("--audit", type=Path, help="Audit saved results without a GPU")
    args = parser.parse_args()
    if args.audit:
        audit(args.audit.resolve())
        return
    if args.runtime_dir is None or args.output is None:
        parser.error("--runtime-dir and --output are required for execution")
    runtime = args.runtime_dir.resolve(strict=True)
    libraries = {name: {"path": str((runtime / name).resolve(strict=True)),
                        "sha256": digest(runtime / name)}
                 for name in ("libamdhip64.so", "libhsa-runtime64.so")}
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    (output / "bin").mkdir()
    selected = cases(args.benchmarks_only)
    manifest = {"complete": False, "benchmarks": args.benchmarks_only, "runs": [],
                "libraries": libraries, "runner_sha256": digest(__file__), "sources": {},
                "binaries": {}, "compiler": subprocess.check_output(
                    [args.hipcc, "--version"], env=clean_environment(), text=True)}
    for kind in sorted({case["kind"] for case in selected}):
        source = SOURCES[kind]
        manifest["sources"][source] = digest(ROOT / source)
        command = [args.hipcc, "-O3" if kind == "waiter" else "-O2",
                   "-std=c++17", "-pthread", "--offload-arch=" + args.gpu_arch,
                   str(ROOT / source), "-o", str(output / "bin" / kind)]
        subprocess.run(command, env=clean_environment(), check=True, timeout=120)
        manifest["binaries"][kind] = digest(output / "bin" / kind)
    for index, case in enumerate(selected):
        kind = case["kind"]
        name = f"{index:02d}-{kind}"
        controls = {"GPU_MAX_HW_QUEUES": str(case.get("queues", 4)),
                    "GPU_GRAPH_DIAGNOSTIC_QUEUE_TRACE": str(int(not args.benchmarks_only)),
                    "GPU_GRAPH_DIAGNOSTIC_FRONTIER_MAX_GENERATIONS": str(case.get("capacity", 1024)),
                    "GPU_GRAPH_DIAGNOSTIC_LANE_FAIL_AFTER": str(case.get("fault", 0)),
                    "DEBUG_CLR_MAX_BATCH_SIZE": "1000"}
        if args.benchmarks_only:
            controls = {"GPU_MAX_HW_QUEUES": "4", "GPU_GRAPH_DIAGNOSTIC_QUEUE_TRACE": "0"}
        if case.get("legacy_zeroed"):
            controls.update({key: "0" for key in LEGACY_CONTROLS})
        env = clean_environment()
        env.update(controls, LD_LIBRARY_PATH=str(runtime) + ":/opt/rocm/lib", LD_DEBUG="libs",
                   LD_PRELOAD=":".join(info["path"] for info in libraries.values()))
        argv = [str(output / "bin" / kind)]
        if kind == "lane":
            argv.append(str(int(bool(case.get("fault")))))
        with (output / (name + ".stdout")).open("w") as stdout, \
                (output / (name + ".stderr")).open("w") as stderr:
            result = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, timeout=120)
        run = {"name": name, "case": case, "controls": controls, "returncode": result.returncode,
               "files": {name + suffix: digest(output / (name + suffix))
                         for suffix in (".stdout", ".stderr")}}
        manifest["runs"].append(run)
        write_json(output / "manifest.json", manifest)
        assert result.returncode == 0, name
        stdout = (output / (name + ".stdout")).read_text()
        stderr = (output / (name + ".stderr")).read_text()
        for info in libraries.values():
            assert digest(info["path"]) == info["sha256"]
        run["mapped"] = mapped_libraries(stderr, libraries)
        run["observed"] = inspect_result(case, stdout, stderr, args.benchmarks_only)
        write_json(output / "manifest.json", manifest)
        print(name, "PASS", flush=True)
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    audit(output)


if __name__ == "__main__":
    main()
