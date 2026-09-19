#!/usr/bin/env python3
"""Stage and run the fixed serving experiment from a Slurm login shell."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONTRACT = json.loads((REPO / "contracts/configuration.json").read_text())
PATH_FIELDS = (
    "shared_root",
    "output_root",
    "sglang_root",
    "image",
    "runtime_archive",
    "runtime_release",
    "model",
    "warm_cache_archive",
)
QUEUE_OVERRIDES = (
    "GPU_MAX_HW_QUEUES",
    "GPU_STREAMOPS_CP_WAIT",
    "AMD_SERIALIZE_KERNEL",
    "AMD_SERIALIZE_COPY",
    "HIP_LAUNCH_BLOCKING",
    "CUDA_LAUNCH_BLOCKING",
    "HSA_ENABLE_SDMA",
    "HSA_CU_MASK",
    "ROC_GLOBAL_CU_MASK",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def site_config(path: Path) -> dict:
    site = json.loads(path.read_text())
    require(
        set(site) == set(PATH_FIELDS) | {"partition", "job_name"}, "Wrong site keys"
    )
    for key in PATH_FIELDS:
        value = site[key]
        require(
            isinstance(value, str) and Path(value).is_absolute(),
            f"{key}: absolute path required",
        )
        require("REQUIRED" not in value, f"Fill the site input: {key}")
        require(
            not any(char in value for char in ",:\n\r\0"), f"Unsafe mount path: {key}"
        )
        site[key] = str(Path(value).resolve())
    shared = Path(site["shared_root"])
    require(shared != Path("/"), "Do not bind the host filesystem root")
    for key in PATH_FIELDS[1:]:
        require(
            Path(site[key]).is_relative_to(shared), f"{key} must be below shared_root"
        )
    output = Path(site["output_root"])
    for key in ("sglang_root", "runtime_release", "model"):
        other = Path(site[key])
        require(
            not output.is_relative_to(other) and not other.is_relative_to(output),
            "Output and input trees must be disjoint",
        )
    for key in ("partition", "job_name"):
        require(
            isinstance(site[key], str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", site[key]) is not None,
            f"Invalid {key}",
        )
        require("REQUIRED" not in site[key], f"Fill {key}")
    return site


def command(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=True, text=True, **kwargs)


def verify_source(source: Path) -> None:
    head = command(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True
    ).stdout.strip()
    require(head == CONTRACT["sglang_base"], "Wrong SGLang base commit")
    for relative, expected in CONTRACT["qualified_files"].items():
        require(
            sha(source / relative) == expected, f"Qualified source mismatch: {relative}"
        )
    status = command(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
    ).stdout
    require(
        {line[3:] for line in status.splitlines()} == set(CONTRACT["qualified_files"]),
        "SGLang has unrelated changes or missing patch files",
    )


def prepare_source(base_repo: Path, destination: Path) -> None:
    require(not destination.exists(), "Source destination already exists")
    command(
        [
            "git",
            "-C",
            str(base_repo),
            "worktree",
            "add",
            "--detach",
            str(destination),
            CONTRACT["sglang_base"],
        ]
    )
    command(
        [
            "git",
            "-C",
            str(destination),
            "apply",
            "--check",
            str(REPO / "patches/sglang-indexer-events.patch"),
        ]
    )
    command(
        [
            "git",
            "-C",
            str(destination),
            "apply",
            str(REPO / "patches/sglang-indexer-events.patch"),
        ]
    )
    verify_source(destination)


def slots() -> list[tuple[str, str]]:
    return [(f"aa-{i:02d}-A", "A") for i in range(8)] + [
        (f"block-{block:02d}-{arm}", arm)
        for block, order in enumerate(CONTRACT["prospective_protocol"]["orders"])
        for arm in order
    ]


def stage(site: dict) -> Path:
    root = Path(site["output_root"])
    require(
        not root.exists(), "Output root already exists; no overwrite or inferred resume"
    )
    verify_source(Path(site["sglang_root"]))
    for key in PATH_FIELDS:
        if key != "output_root":
            require(Path(site[key]).exists(), f"Missing site input: {key}")
    root.mkdir(parents=True, exist_ok=False)
    (root / "phase2").mkdir()
    (root / "phase2/source").symlink_to(site["sglang_root"], target_is_directory=True)
    shutil.copytree(
        REPO / "harness", root / "phase3", ignore=shutil.ignore_patterns("__pycache__")
    )
    (root / "phase4").mkdir()
    shutil.copytree(REPO / "contracts", root / "contracts")
    for name in ("reproduce.py", "worker.py", "analyze.py"):
        shutil.copyfile(REPO / name, root / name)
    # The two reviewed deployment substitutions are already in the payload.
    changes = json.loads((REPO / "contracts/relocations.json").read_text())
    for change in changes:
        staged = root / "phase3" / Path(change["file"]).name
        require(sha(staged) == change["portable_sha256"], "Portable payload changed")
    argv = json.loads((REPO / "contracts/server-argv.json").read_text())
    argv[argv.index("--model-path") + 1] = site["model"]
    write_json(root / "phase3/server-argv.json", argv)
    write_json(root / "site.json", site)
    write_json(root / "relocations.json", changes)
    files = {}
    for folder in ("phase3", "contracts"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(root))] = sha(path)
    for relative, expected in CONTRACT["qualified_files"].items():
        files["phase2/source/" + relative] = expected
    for name in (
        "reproduce.py",
        "worker.py",
        "analyze.py",
        "site.json",
        "relocations.json",
    ):
        files[name] = sha(root / name)
    write_json(
        root / "execution-manifest.json",
        dict(
            execution_files=files,
            protocol=CONTRACT["prospective_protocol"],
            qualification="Portable orchestration CPU-tested; inherited task-only numerical qualification, no production or full-model equivalence guarantee",
        ),
    )
    return root


def seconds_left(value: str) -> int:
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    parts = clock.split(":")
    require(
        days.isdigit() and len(parts) in (2, 3) and all(p.isdigit() for p in parts),
        "Unrecognized bounded Slurm time",
    )
    return int(days) * 86400 + sum(
        int(p) * 60**i for i, p in enumerate(reversed(parts))
    )


def owned_job(job: int, site: dict) -> dict:
    require(job > 0, "A positive explicit job ID is required")
    text = command(
        ["scontrol", "show", "job", str(job), "-o"], capture_output=True, timeout=15
    ).stdout
    fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", text))
    require(fields.get("JobId") == str(job), "Wrong Slurm job")
    require(
        fields.get("UserId") == f"{getpass.getuser()}({os.getuid()})",
        "Job is not owned by this user",
    )
    require(
        fields.get("JobName") == site["job_name"]
        and fields.get("Partition") == site["partition"],
        "Job name/partition mismatch",
    )
    require(
        fields.get("JobState") == "RUNNING"
        and fields.get("NumNodes") == "1"
        and fields.get("NumCPUs") == "64",
        "Require one running node with 64 CPUs",
    )
    tres = fields.get("AllocTRES", fields.get("TRES", ""))
    require(
        re.search(r"(?:^|,)gres/gpu=8(?:,|$)", tres) is not None,
        "Require exactly eight allocated GPUs",
    )
    require(
        seconds_left(fields["TimeLimit"]) <= 21600,
        "Allocation exceeds the declared six-hour envelope",
    )
    left = command(
        ["squeue", "-j", str(job), "-h", "-o", "%L"], capture_output=True, timeout=15
    ).stdout.strip()
    return dict(raw=text, seconds_left=seconds_left(left), node=fields["NodeList"])


def step_prefix(
    job: int, site: dict, *, container: bool, create: bool = False
) -> list[str]:
    argv = ["srun", f"--jobid={job}", "--overlap", "--ntasks=1", "--cpus-per-task=64"]
    if container:
        cache = f"/tmp/native-runtime-reproducer-{job}"
        argv += [
            f"--container-name=native-runtime-reproducer-{job}",
            "--container-writable",
            f"--container-mounts={site['shared_root']}:{site['shared_root']},{cache}:/native-runtime-cache,{cache}-claim:{cache}-claim,{cache}/aiter-jit:/sgl-workspace/aiter/aiter/jit,tmpfs:/dev/shm",
            "--no-container-entrypoint",
            "--no-container-remap-root",
            "--no-container-mount-home",
        ]
        if create:
            argv.append(f"--container-image={site['image']}")
    return argv


def admit(root: Path, name: str, returncode: int, log: Path) -> tuple[dict, dict]:
    require(returncode == 0, f"Nonzero serving step: {returncode}")
    lines = log.read_text().splitlines()
    emitted = [
        json.loads(line.removeprefix("NATIVE_GROUP_DONE "))
        for line in lines
        if line.startswith("NATIVE_GROUP_DONE ")
    ]
    require(
        len(emitted) == 1 and emitted[0].get("valid") is True,
        "Missing, duplicate or invalid emitted receipt",
    )
    require("NATIVE_STEP_DONE rc=0" in lines, "No successful wrapper receipt")
    identities = [
        json.loads(line.removeprefix("NATIVE_GROUP_IDENTITY "))
        for line in lines
        if line.startswith("NATIVE_GROUP_IDENTITY ")
    ]
    identity_path = root / "phase4" / name / "group-identity.json"
    require(
        len(identities) == 1
        and identities[0] == json.loads(identity_path.read_text())
        and identities[0]["name"] == name
        and identities[0]["arm"] == name[-1],
        "Emitted/disk group identity mismatch",
    )
    start = time.monotonic()
    while True:
        try:
            status = json.loads((root / "phase4" / name / "status.json").read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            require(time.monotonic() - start < 60, "Receipt visibility timeout")
            time.sleep(0.25)
    require(status == emitted[0], "Emitted/disk receipt mismatch")
    require(
        status.get("cleanup") == [] and not status.get("worker_cleanup_survivors"),
        "Worker cleanup failed",
    )
    require(
        status.get("work_budget_seconds") == 495
        and status.get("cleanup_reservation_seconds") == 15
        and status["total_seconds"] <= 510,
        "Wrong or exceeded prospective cap",
    )
    source = json.loads(
        (root / "phase4" / name / "source-attestation.json").read_text()
    )
    manifest = json.loads((root / "execution-manifest.json").read_text())
    require(
        source.get("valid")
        and source.get("manifest_sha256") == sha(root / "execution-manifest.json")
        and source.get("verified_files") == manifest["execution_files"],
        "Source binding mismatch",
    )
    return status, dict(seconds=time.monotonic() - start, emitted_disk_exact_match=True)


def verify_claim_visibility(root: Path, expected: dict) -> None:
    deadline = time.monotonic() + 60
    while True:
        try:
            receipt = json.loads((root / "claim.json").read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            require(time.monotonic() < deadline, "Claim receipt visibility timeout")
            time.sleep(0.25)
    require(receipt == expected, "Emitted/disk claim receipt mismatch")


def run(site: dict, job: int) -> None:
    require(
        not os.environ.get("SLURM_JOB_ID"),
        "Run Slurm control from the login shell outside a job step",
    )
    require(
        not any(key in os.environ for key in QUEUE_OVERRIDES),
        "Inherited GPU queue/serialization override",
    )
    receipt = owned_job(job, site)
    root = stage(site)
    claim_request = dict(
        job_id=str(job), root=str(root), uid=os.getuid(), token=secrets.token_hex(24)
    )
    write_json(root / "claim-request.json", claim_request)
    # This is outside the release-cleanup region. A losing claimant must never
    # cancel another controller's allocation, even on a different login host.
    claim_result = command(
        [
            *step_prefix(job, site, container=False),
            "python3",
            str(root / "worker.py"),
            "claim",
            str(root),
        ],
        capture_output=True,
        timeout=60,
    )
    emitted_claims = [
        json.loads(line.removeprefix("NATIVE_CLAIM_DONE "))
        for line in claim_result.stdout.splitlines()
        if line.startswith("NATIVE_CLAIM_DONE ")
    ]
    require(
        emitted_claims == [claim_request],
        "Missing or mismatched successful node claim receipt",
    )
    # Authority is established once, before the cleanup region. A later node
    # RPC failure cannot revoke this controller's duty to release this exact job.
    acquired_claim_sha256 = hashlib.sha256(
        (json.dumps(claim_request, indent=2) + "\n").encode()
    ).hexdigest()
    record = dict(
        job_id=job,
        started_unix=time.time(),
        groups=[],
        invalid_attempts=[],
        orders=CONTRACT["prospective_protocol"]["orders"],
        protocol="prospective_uniform_510_seconds",
        descriptive_only=True,
        accepted=False,
        completed=False,
        initial_allocation=receipt,
    )
    campaign_path = root / "phase4/campaign.json"
    write_json(campaign_path, record)
    created = False
    active = None
    try:
        (root / "claim.log").write_text(claim_result.stdout + claim_result.stderr)
        verify_claim_visibility(root, claim_request)
        with (root / "prepare.log").open("x") as log:
            command(
                [
                    *step_prefix(job, site, container=False),
                    "python3",
                    str(root / "worker.py"),
                    "prepare",
                    str(root),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        with (root / "bootstrap.log").open("x") as log:
            command(
                [
                    *step_prefix(job, site, container=True, create=True),
                    "python3",
                    str(root / "worker.py"),
                    "bootstrap",
                    str(root),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        created = True
        for index, (name, arm) in enumerate(slots()):
            state = owned_job(job, site)
            require(
                state["seconds_left"]
                >= CONTRACT["prospective_protocol"]["allocation_admission_seconds"],
                "Insufficient allocation time for another bounded group",
            )
            active = name
            start = time.monotonic()
            log_path = root / "phase4" / f"{name}-step.log"
            with log_path.open("x") as log:
                process = subprocess.run(
                    [
                        *step_prefix(job, site, container=True),
                        "python3",
                        str(root / "worker.py"),
                        "group",
                        str(root),
                        name,
                        arm,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=690,
                )
            status, visibility = admit(root, name, process.returncode, log_path)
            identity = json.loads(
                (root / "phase4" / name / "group-identity.json").read_text()
            )
            require(
                identity["job_id"] == str(job)
                and identity["name"] == name
                and identity["arm"] == arm,
                "Wrong serving-step identity",
            )
            row = dict(
                name=name,
                arm=arm,
                serving_step_id=identity["step_id"],
                step_rc=process.returncode,
                wall_seconds=time.monotonic() - start,
                status=status,
                receipt_visibility=visibility,
            )
            record["groups"].append(row)
            write_json(campaign_path, record)
            print(
                json.dumps(
                    dict(name=name, valid=True, total_seconds=status["total_seconds"])
                ),
                flush=True,
            )
            active = None
            if index == 7:
                with (root / "calibration.log").open("x") as log:
                    command(
                        [
                            *step_prefix(job, site, container=True),
                            "python3",
                            str(root / "phase3/analyze.py"),
                            str(root / "phase4"),
                            "--calibrate",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                record["calibration_sha256"] = sha(root / "phase4/calibration.json")
                write_json(campaign_path, record)
        record["completed"] = True
    except BaseException as error:
        record["error"] = repr(error)
        if active:
            path = root / "phase4" / active / "status.json"
            status = json.loads(path.read_text()) if path.exists() else None
            identity_path = root / "phase4" / active / "group-identity.json"
            identity = (
                json.loads(identity_path.read_text()) if identity_path.exists() else {}
            )
            record["invalid_attempts"].append(
                dict(
                    name=active,
                    arm=active[-1],
                    serving_step_id=identity.get("step_id"),
                    status=status,
                    replaced=False,
                )
            )
        raise
    finally:
        record["ended_unix"] = time.time()
        write_json(campaign_path, record)
        cleanup = dict(
            job_id=job,
            health_verified=False,
            cache_absent_verified=False,
            released_verified=False,
            acquired_claim_sha256=acquired_claim_sha256,
        )
        if created:
            try:
                with (root / "health.log").open("x") as log:
                    result = subprocess.run(
                        [
                            *step_prefix(job, site, container=True),
                            "python3",
                            str(root / "worker.py"),
                            "health",
                            str(root),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                cleanup["health_step_returncode"] = result.returncode
                cleanup["health_verified"] = result.returncode == 0
            except (Exception, KeyboardInterrupt) as error:
                cleanup["health_error"] = repr(error)
        # Preparation may have created owned cache before failing. A health
        # timeout must not skip this independent, ownership-checked attempt.
        try:
            with (root / "cache-cleanup.log").open("x") as log:
                result = subprocess.run(
                    [
                        *step_prefix(job, site, container=False),
                        "python3",
                        str(root / "worker.py"),
                        "cleanup",
                        str(root),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
            cleanup["cache_cleanup_returncode"] = result.returncode
            cache_receipt = root / "cache-cleanup.json"
            cleanup["cache_absent_verified"] = (
                result.returncode == 0
                and cache_receipt.exists()
                and json.loads(cache_receipt.read_text()).get("absent") is True
            )
        except (Exception, KeyboardInterrupt) as error:
            cleanup["cache_cleanup_error"] = repr(error)
        finally:
            try:
                command(
                    [
                        *step_prefix(job, site, container=False),
                        "python3",
                        str(root / "worker.py"),
                        "authorize-release",
                        str(root),
                    ],
                    capture_output=True,
                    timeout=60,
                )
                cleanup["claim_recheck_verified"] = True
            except (Exception, KeyboardInterrupt) as error:
                cleanup["claim_recheck_error"] = repr(error)
                cleanup["release_authority"] = "verified initial atomic claim"
            try:
                result = subprocess.run(
                    ["scancel", str(job)], capture_output=True, text=True, timeout=15
                )
                cleanup["cancel_returncode"] = result.returncode
            except (Exception, KeyboardInterrupt) as error:
                cleanup["cancel_error"] = repr(error)
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    queues = [
                        subprocess.run(args, capture_output=True, text=True, timeout=15)
                        for args in (
                            ["squeue", "--jobs", str(job), "--noheader"],
                            ["squeue", "--steps", "--jobs", str(job), "--noheader"],
                        )
                    ]
                    if all(
                        row.returncode == 0 and not row.stdout.strip() for row in queues
                    ):
                        cleanup["released_verified"] = True
                        break
                    time.sleep(2)
            except (Exception, KeyboardInterrupt) as error:
                cleanup["queue_verification_error"] = repr(error)
            try:
                accounting = subprocess.run(
                    [
                        "sacct",
                        "-j",
                        str(job),
                        "--parsable2",
                        "--format=JobID,JobName,User,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                (root / "accounting.txt").write_text(
                    accounting.stdout + accounting.stderr
                )
                cleanup["accounting_returncode"] = accounting.returncode
            except (Exception, KeyboardInterrupt) as error:
                cleanup["accounting_error"] = repr(error)
            finally:
                write_json(root / "release.json", cleanup)
    require(
        record["completed"]
        and cleanup["released_verified"]
        and cleanup["health_verified"]
        and cleanup["cache_absent_verified"],
        "Campaign or owned-resource cleanup was not fully verified",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    source = commands.add_parser("prepare-source")
    source.add_argument("--base-repo", type=Path, required=True)
    source.add_argument("--destination", type=Path, required=True)
    for name in ("plan", "run"):
        item = commands.add_parser(name)
        item.add_argument("--site", type=Path, required=True)
        if name == "run":
            item.add_argument("--job-id", type=int, required=True)
    args = parser.parse_args()
    if args.action == "prepare-source":
        prepare_source(args.base_repo.resolve(), args.destination.resolve())
    elif args.action == "plan":
        site = site_config(args.site)
        require(not Path(site["output_root"]).exists(), "Output root is not fresh")
        print(
            json.dumps(
                dict(
                    site=site, slots=slots(), protocol=CONTRACT["prospective_protocol"]
                ),
                indent=2,
            )
        )
    else:
        run(site_config(args.site), args.job_id)


if __name__ == "__main__":
    main()
