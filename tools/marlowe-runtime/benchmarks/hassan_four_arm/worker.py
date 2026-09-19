"""Bounded node work; Slurm control stays in the login-side controller."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from reproduce import CONTRACT, require, sha, verify_source, write_json


def cache_path() -> Path:
    job = os.environ["SLURM_JOB_ID"]
    require(job.isdigit(), "Missing numeric allocated job ID")
    return Path(f"/tmp/native-runtime-reproducer-{job}")


def claim_path() -> Path:
    return cache_path().with_name(cache_path().name + "-claim")


def claim(root: Path) -> None:
    request = json.loads((root / "claim-request.json").read_text())
    require(
        request["job_id"] == os.environ["SLURM_JOB_ID"]
        and request["uid"] == os.getuid()
        and request["root"] == str(root),
        "Wrong claim request",
    )
    path = claim_path()
    path.mkdir(mode=0o700, exist_ok=False)
    write_json(path / "owner.json", request)
    write_json(root / "claim.json", request)
    print("NATIVE_CLAIM_DONE " + json.dumps(request), flush=True)


def require_claim(root: Path) -> None:
    path = claim_path()
    require(
        path.is_dir() and not path.is_symlink() and path.stat().st_uid == os.getuid(),
        "Missing owned job claim",
    )
    owner = json.loads((path / "owner.json").read_text())
    request = json.loads((root / "claim-request.json").read_text())
    require(
        owner == request == json.loads((root / "claim.json").read_text())
        and owner["job_id"] == os.environ["SLURM_JOB_ID"]
        and owner["root"] == str(root),
        "This controller does not own release authority",
    )


def verify_runtime_release(release: Path, runtime: dict) -> None:
    require(
        json.loads((release / "manifest.json").read_text()) == runtime,
        "Runtime release manifest differs from the measured artifact",
    )
    for name, expected in runtime["libraries"].items():
        require(
            sha(release / "lib" / name) == expected, f"Wrong runtime library: {name}"
        )
    for alias, target in runtime["aliases"].items():
        require(
            (release / "lib" / alias).is_symlink()
            and (release / "lib" / alias).readlink() == Path(target),
            f"Wrong runtime alias: {alias}",
        )
    # assemble.py installs these two launch files. Other recipe hashes describe
    # the build inputs; they are not files at the release root.
    for name in ("run", "verify.py"):
        require(
            sha(release / name) == runtime["recipe_sha256"][name],
            f"Wrong runtime launch file: {name}",
        )


def preflight(site: dict) -> dict:
    verified = {}
    for key, expected_key in (
        ("image", "image_sha256"),
        ("runtime_archive", "runtime_archive_sha256"),
        ("warm_cache_archive", "warm_cache_archive_sha256"),
    ):
        actual = sha(Path(site[key]))
        require(actual == CONTRACT[expected_key], f"Wrong {key} bytes")
        verified[key] = actual
    verify_source(Path(site["sglang_root"]))
    release = Path(site["runtime_release"])
    runtime = json.loads((Path(__file__).parent / "contracts/runtime.json").read_text())
    verify_runtime_release(release, runtime)
    models = json.loads(
        (Path(__file__).parent / "contracts/model-files.json").read_text()
    )
    for row in models:
        path = Path(site["model"]) / row["name"]
        require(
            path.is_file()
            and path.stat().st_size == row["bytes"]
            and sha(path) == row["sha256"],
            f"Model file mismatch: {row['name']}",
        )
    verified["model_files"] = len(models)
    verified["source"] = CONTRACT["qualified_files"]
    return verified


def prepare(root: Path, site: dict) -> None:
    require_claim(root)
    cache = cache_path()
    require(not cache.exists(), "Owned node cache already exists")
    receipt = preflight(site)
    write_json(root / "asset-verification.json", receipt)
    cache.mkdir()
    write_json(
        cache / "owner.json",
        dict(job_id=os.environ["SLURM_JOB_ID"], root=str(root), uid=os.getuid()),
    )
    with tarfile.open(site["warm_cache_archive"]) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and path.parts[0] == "seed",
                "Unsafe cache archive path",
            )
            require(
                member.isfile() or member.isdir(),
                "Unexpected cache archive link/device",
            )
        # Every member was checked before extraction; this also supports the
        # Python 3.10 host interpreter used by some Slurm login/compute images.
        archive.extractall(cache)
    require(
        sha(cache / "seed/inventory.json") == CONTRACT["cache_inventory_sha256"],
        "Wrong cache inventory",
    )
    for name in ("working-cache", "aiter-jit"):
        shutil.copytree(
            cache / "seed" / name, cache / name, copy_function=shutil.copyfile
        )


def bootstrap(root: Path) -> None:
    import torch

    require(
        os.environ.get("SLURM_GPUS_ON_NODE") == "8", "Step allocation is not eight GPUs"
    )
    require(torch.cuda.device_count() == 8, "Actual device visibility is not eight")
    require(
        sorted(os.sched_getaffinity(0)) == CONTRACT["expected_affinity"],
        "CPU affinity differs from the recorded two-socket placement",
    )
    arch = [torch.cuda.get_device_properties(i).gcnArchName for i in range(8)]
    require(
        all(value.split(":")[0] == "gfx950" for value in arch), "Wrong GPU architecture"
    )
    write_json(
        root / "bootstrap.json",
        dict(
            valid=True,
            job_id=os.environ["SLURM_JOB_ID"],
            architecture=arch,
            affinity=sorted(os.sched_getaffinity(0)),
            torch=torch.__version__,
            hip=torch.version.hip,
        ),
    )


def attest(root: Path, output: Path) -> None:
    path = root / "execution-manifest.json"
    files = json.loads(path.read_text())["execution_files"]
    for name, expected in files.items():
        require(sha(root / name) == expected, f"Staged source changed: {name}")
    write_json(
        output / "source-attestation.json",
        dict(valid=True, manifest_sha256=sha(path), verified_files=files),
    )


def validate_workers(output: Path, arm: str, step_id: str) -> list[int]:
    records = []
    require(
        isinstance(step_id, str) and step_id.isdigit(),
        "Missing numeric serving step identity",
    )
    expected_names = {
        f"worker-{role}-rank{rank}.json"
        for role in ("target", "draft")
        for rank in range(8)
    }
    require(
        {path.name for path in output.glob("worker-*-rank*.json")} == expected_names,
        "Wrong worker filename census",
    )
    expected_dispatch = json.loads(
        (Path(__file__).parent / "contracts/projection-dispatch.json").read_text()
    )
    for role in ("target", "draft"):
        rows = []
        for path in output.glob(f"worker-{role}-rank*.json"):
            row = json.loads(path.read_text())
            rank = int(
                re.fullmatch(
                    r"worker-(?:target|draft)-rank([0-7])\.json", path.name
                ).group(1)
            )
            require(
                row["rank"] == rank and row["role"] == role,
                "Filename/worker identity mismatch",
            )
            rows.append(row)
        require(
            sorted(row["rank"] for row in rows) == list(range(8)),
            f"Incomplete {role} census",
        )
        require(
            len({row["pid"] for row in rows}) == 8
            and all(
                isinstance(row["pid"], int)
                and not isinstance(row["pid"], bool)
                and row["pid"] > 0
                for row in rows
            ),
            "Require eight distinct valid PIDs per role",
        )
        for row in rows:
            require(
                row["role"] == role
                and row["env"]["SLURM_JOB_ID"] == os.environ["SLURM_JOB_ID"],
                "Wrong worker identity",
            )
            require(
                row["env"]["NATIVE_CORRECTNESS"] == row["env"]["NATIVE_PROFILE"] == "0",
                "Diagnostic primary worker",
            )
            require(
                row["env"]["SLURM_STEP_ID"] == step_id
                and row["env"]["SLURM_GPUS_ON_NODE"] == "8",
                "Stale/mixed serving-step receipt",
            )
            for key, value in CONTRACT["arms"][arm].items():
                require(row["env"][key] == value, "Wrong arm toggle")
            require(row["runtime"]["gpu_count"] == 8, "Wrong runtime device count")
            if role == "target":
                require(
                    row["weight_bytes"] == CONTRACT["target_weight_bytes_per_rank"],
                    "Wrong retained target bytes",
                )
                require(
                    len(row["layers"]) == 10
                    and row["invoked_indexers"] == [0, 1, 2, 6],
                    "Wrong target/indexer depth",
                )
                require(
                    row["selected_projection_dispatch"] == expected_dispatch,
                    "Projection dispatch changed",
                )
            else:
                require(
                    len(row["layers"]) == 1
                    and row["decode_runner_class"] == "EagerRunner",
                    "Wrong retained draft role/mode",
                )
            records.append(row)
    return sorted({row["pid"] for row in records})


def smi(path: Path) -> None:
    result = subprocess.run(
        [
            "rocm-smi",
            "--showmemuse",
            "--showmeminfo",
            "vram",
            "--showuse",
            "--showpids",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    path.write_text(result.stdout + result.stderr)


def group(root: Path, site: dict, name: str, arm: str) -> int:
    require_claim(root)
    require(
        re.fullmatch(r"(?:aa-0[0-7]-A|block-0[0-7]-[ABCD])", name) is not None
        and name[-1] == arm,
        "Invalid scheduled slot",
    )
    output = root / "phase4" / name
    require(not output.exists(), "Refuse replacing an attempt")
    env = dict(
        os.environ,
        NATIVE_ROOT=str(root),
        NATIVE_ARM_OUTPUT=str(output),
        NATIVE_CACHE="/native-runtime-cache/working-cache",
        NATIVE_MODEL_PATH=site["model"],
        NATIVE_RUNTIME_RELEASE=site["runtime_release"],
        NATIVE_CORRECTNESS="0",
        NATIVE_PROFILE="0",
        **CONTRACT["arms"][arm],
    )
    subprocess.run(
        [sys.executable, str(root / "phase3/reset_cache.py"), str(output)],
        env=env,
        check=True,
        timeout=60,
    )
    identity = dict(
        job_id=os.environ["SLURM_JOB_ID"],
        step_id=os.environ["SLURM_STEP_ID"],
        name=name,
        arm=arm,
    )
    write_json(output / "group-identity.json", identity)
    print("NATIVE_GROUP_IDENTITY " + json.dumps(identity), flush=True)
    attest(root, output)
    smi(output / "health-before.txt")
    script = 'source "$NATIVE_ROOT/phase3/common_env.sh"\nexec "$NATIVE_RUNTIME_RELEASE/run" python3 "$NATIVE_ROOT/phase3/run_group.py" --kind performance --cap-seconds 510\n'
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script], env=env, timeout=540
    )
    smi(output / "health-after.txt")
    if result.returncode == 0:
        validate_workers(output, arm, identity["step_id"])
    print(f"NATIVE_STEP_DONE rc={result.returncode}", flush=True)
    return result.returncode


def parsed_health(text: str) -> dict:
    """Admit only the recognized memory-section reorder, keeping strict KFD rows."""
    parent = Path(__file__).parent
    path = (
        parent
        / ("phase3" if (parent / "phase3").exists() else "harness")
        / "parse_health.py"
    )
    spec = importlib.util.spec_from_file_location("strict_smi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = text.splitlines(keepends=True)
    positions = []
    for title in ("KFD Processes", "Memory Usage (Bytes)", "End of ROCm SMI Log"):
        hits = [i for i, line in enumerate(lines) if title in line]
        require(len(hits) == 1, "Missing/repeated SMI section")
        positions.append(hits[0])
    kfd, memory, end = positions
    if kfd < memory < end:
        for line in lines[memory + 1 : end]:
            stripped = line.strip()
            require(
                not stripped
                or set(stripped) == {"="}
                or re.fullmatch(
                    r"GPU\[[0-7]\]\s*: VRAM Total (?:Used )?Memory \(B\): \d+", stripped
                )
                is not None,
                "Unrecognized memory section content",
            )
        text = "".join(
            lines[:kfd] + lines[memory:end] + lines[kfd:memory] + lines[end:]
        )
    return module.parse_health(text)


def health(root: Path) -> None:
    campaign = json.loads((root / "phase4/campaign.json").read_text())
    require(
        campaign["job_id"] == int(os.environ["SLURM_JOB_ID"])
        and "ended_unix" in campaign,
        "Wrong or live campaign",
    )
    records = [*campaign["groups"], *campaign["invalid_attempts"]]
    require(len({row["name"] for row in records}) == len(records), "Duplicate attempt")
    folders = {
        path.name
        for path in (root / "phase4").iterdir()
        if path.is_dir() and path.name.startswith(("aa-", "block-"))
    }
    require(folders == {row["name"] for row in records}, "Incomplete attempt coverage")
    pids = set()
    steps = set()
    for row in records:
        step_id = row.get("serving_step_id")
        require(step_id not in steps, "Serving step reused across attempts")
        steps.add(step_id)
        output = root / "phase4" / row["name"]
        identity = json.loads((output / "group-identity.json").read_text())
        require(
            identity
            == dict(
                job_id=str(campaign["job_id"]),
                step_id=step_id,
                name=row["name"],
                arm=row["arm"],
            ),
            "Terminal group identity mismatch",
        )
        pids.update(validate_workers(output, row["arm"], step_id))
    require(
        all(not Path(f"/proc/{pid}").exists() for pid in pids),
        "Recorded worker PID still exists; no inferred reuse",
    )
    path = root / "health-final.txt"
    smi(path)
    data = parsed_health(path.read_text())
    require(
        data["gpu_allocated_vram_percent"] == dict.fromkeys(range(8), 0)
        and data["gpu_busy_percent"] == dict.fromkeys(range(8), 0)
        and all(row["vram_bytes"] == 0 for row in data["kfd_processes"]),
        "GPU resources are not idle",
    )
    write_json(
        root / "health-final.json",
        dict(
            valid=True,
            job_id=campaign["job_id"],
            distinct_worker_pids=len(pids),
            all_recorded_workers_absent=True,
            **data,
        ),
    )


def cleanup(root: Path) -> None:
    require_claim(root)
    cache = cache_path()
    campaign = json.loads((root / "phase4/campaign.json").read_text())
    if campaign["groups"] or campaign["invalid_attempts"]:
        require(
            (root / "health-final.json").exists()
            and json.loads((root / "health-final.json").read_text()).get("valid")
            is True,
            "Cache cleanup requires verified stopped workers",
        )
    if not cache.exists():
        write_json(
            root / "cache-cleanup.json",
            dict(job_id=os.environ["SLURM_JOB_ID"], cache=str(cache), absent=True),
        )
        return
    require(
        cache.is_dir()
        and not cache.is_symlink()
        and cache.stat().st_uid == os.getuid(),
        "Cache ownership mismatch",
    )
    owner = json.loads((cache / "owner.json").read_text())
    require(
        owner
        == dict(job_id=os.environ["SLURM_JOB_ID"], root=str(root), uid=os.getuid()),
        "Wrong cache owner receipt",
    )
    require(
        set(path.name for path in cache.iterdir())
        <= {"owner.json", "seed", "working-cache", "aiter-jit"},
        "Unexpected owned cache content",
    )
    shutil.rmtree(cache)
    write_json(
        root / "cache-cleanup.json",
        dict(
            job_id=os.environ["SLURM_JOB_ID"],
            cache=str(cache),
            absent=not cache.exists(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "claim",
            "authorize-release",
            "prepare",
            "bootstrap",
            "group",
            "health",
            "cleanup",
        ),
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("name", nargs="?")
    parser.add_argument("arm", nargs="?")
    args = parser.parse_args()
    site = json.loads((args.root / "site.json").read_text())
    if args.action == "claim":
        claim(args.root)
    elif args.action == "authorize-release":
        require_claim(args.root)
    elif args.action == "prepare":
        prepare(args.root, site)
    elif args.action == "bootstrap":
        bootstrap(args.root)
    elif args.action == "group":
        raise SystemExit(group(args.root, site, args.name, args.arm))
    elif args.action == "health":
        health(args.root)
    else:
        cleanup(args.root)
