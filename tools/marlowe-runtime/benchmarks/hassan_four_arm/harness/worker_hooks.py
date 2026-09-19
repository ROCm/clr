"""Private worker provenance and opt-in correctness; no primary timer patch."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch


def _write(name, value):
    root = Path(os.environ["NATIVE_ARM_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(value, indent=2, default=str))


def _runtime():
    torch.ones(16, device="cuda").mul_(2)
    torch.cuda.synchronize()
    return runpy.run_path(os.environ["MARLOWE_RUNTIME_ROOT"] + "/verify.py")["verify"]()


def install():
    import aiter.tuned_gemm as tg
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    diagnostic = None
    if os.environ.get("NATIVE_CORRECTNESS") == "1":
        spec = importlib.util.spec_from_file_location(
            "native_correctness",
            os.environ["NATIVE_CORRECTNESS_SOURCE"],
        )
        diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostic)
        diagnostic.install()

    selected = {}
    originals = dict(tg.solMap)
    for name, func in originals.items():

        def observe(*args, _name=name, _func=func, **kwargs):
            x, weight = args[:2]
            shape = (x.shape[0], weight.shape[0], x.shape[-1])
            if shape in ((4, 4096, 2048), (4, 128, 6144)):
                selected[str(shape)] = dict(
                    libtype=_name,
                    solution=args[2],
                    config=kwargs.get("config"),
                    input_dtype=str(x.dtype),
                    weight_dtype=str(weight.dtype),
                    input_stride=x.stride(),
                    weight_stride=weight.stride(),
                )
            return _func(*args, **kwargs)

        tg.solMap[name] = observe

    backend_init = FullCudaGraphBackend.__init__

    def backend(self, runner, *args, **kwargs):
        backend_init(self, runner, *args, **kwargs)
        owner = self._projection_schedule
        if owner is None:
            return
        # The real baseline runner has initialized capture_forward_mode already.
        if not runner.capture_forward_mode.is_target_verify():
            raise AssertionError("Target owner missed capture mode initialization")
        indexers = self._projection_indexers
        configs = []
        for indexer in indexers:
            if not hasattr(indexer, "wq_b") and not hasattr(indexer, "wk"):
                continue
            for name, n, k in (("wq_b", 4096, 2048), ("wk", 128, 6144)):
                projection = getattr(indexer, name)
                if type(projection.quant_method).__name__ != "UnquantizedLinearMethod":
                    raise AssertionError(
                        f"Unexpected quantization dispatch: {projection}"
                    )
                weight = projection.weight
                if list(weight.shape) != [n, k] or weight.dtype != torch.bfloat16:
                    raise AssertionError(
                        f"Unexpected BF16 projection: {name} {weight.shape} {weight.dtype}"
                    )
                configs.append(
                    tg.get_GEMM_A16W16_config(
                        4,
                        n,
                        k,
                        False,
                        str(weight.dtype),
                        str(weight.dtype),
                        False,
                        False,
                    )
                )
        # hipb_gemm uses one writable process-global allocation. Reject concurrent
        # use before capture; no serialization or replacement kernel is introduced.
        if configs[0]["libtype"] == configs[1]["libtype"] == "hipblaslt":
            raise AssertionError(
                f"Concurrent Q/K share AITER hipBLASLt workspace: {configs[:2]}"
            )
        if diagnostic is not None:
            diagnostic.attach(runner.model_runner, indexers)

    FullCudaGraphBackend.__init__ = backend
    if diagnostic is not None:
        original_capture_one = FullCudaGraphBackend.capture_one

        def capture_one(self, shape_key, *args, **kwargs):
            if self._projection_schedule is not None:
                diagnostic.begin_capture()
            original_capture_one(self, shape_key, *args, **kwargs)
            if self._projection_schedule is not None:
                diagnostic.finish_capture(shape_key)

        FullCudaGraphBackend.capture_one = capture_one
    runner_init = ModelRunner.__init__
    bridge_receipt = {}

    def initialize(self, *args, **kwargs):
        self._native_start = time.monotonic()
        if os.environ.get("GPU_NATIVE_EVENT_TRACE") == "1":
            trace_path = (
                Path(os.environ["NATIVE_ARM_OUTPUT"])
                / f"runtime-worker-{os.getpid()}.stderr"
            )
            descriptor = os.open(
                trace_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            os.dup2(descriptor, 2)
            os.close(descriptor)
        runner_init(self, *args, **kwargs)
        if not bridge_receipt:
            from host_pointer_bridge import install as install_host_pointer_bridge

            _runtime()
            bridge_receipt.update(install_host_pointer_bridge())
            _runtime()

    ModelRunner.__init__ = initialize
    graph_init = ModelRunner.init_cuda_graphs

    def initialize_graphs(self, *args, **kwargs):
        # This source defers graph initialization until both models are loaded.
        graph_init(self, *args, **kwargs)
        rank = torch.distributed.get_rank()
        role = "draft" if self.is_draft_worker else "target"
        layers = [
            (name, module.layer_id)
            for name, module in self.model.named_modules()
            if type(module).__name__ == "DeepseekV2DecoderLayer"
        ]
        expected = 1 if self.is_draft_worker else 10
        if len(layers) != expected:
            raise AssertionError(
                f"Rank {rank} {role} instantiated {layers}, expected {expected}"
            )
        value = torch.ones(1, device="cuda")
        torch.distributed.all_reduce(value)
        torch.cuda.synchronize()
        if value.item() != 8:
            raise AssertionError(f"Expected TP8 collective: {value}")
        graph_runner = getattr(self, "decode_cuda_graph_runner", None)
        owner = getattr(
            getattr(graph_runner, "backend", None), "_projection_schedule", None
        )
        if not self.is_draft_worker and owner is None:
            raise AssertionError("Requested target projection owner is absent")
        if diagnostic is not None and owner is not None:
            diagnostic.check_lifecycle(graph_runner)
            owner = graph_runner.backend._projection_schedule
        receipt = dict(
            rank=rank,
            role=role,
            pid=os.getpid(),
            startup_seconds=time.monotonic() - self._native_start,
            runtime=_runtime(),
            layers=layers,
            weight_bytes=sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            ),
            source_model=type(self.model).__module__,
            host_pointer_bridge=bridge_receipt,
            decode_runner_class=type(graph_runner).__name__,
            decode_backend_class=type(getattr(graph_runner, "backend", None)).__name__,
            selected_projection_dispatch=selected,
            graph_capture_mode=str(getattr(graph_runner, "capture_forward_mode", None)),
            graph_keys=[
                str(k)
                for k in getattr(getattr(graph_runner, "backend", None), "_graphs", {})
            ],
            instantiated_indexers=list(owner.layer_ids) if owner else [],
            weighted_indexers=[
                m.layer_id
                for m in graph_runner.backend._projection_indexers
                if hasattr(m, "wq_b") and hasattr(m, "wk")
            ]
            if owner
            else [],
            invoked_indexers=sorted({row[0] for row in owner.census}) if owner else [],
            schedule=owner.mode if owner else None,
            projection_stream=owner.stream.cuda_stream if owner else None,
            capture_stream=graph_runner.stream.cuda_stream if owner else None,
            env={
                key: os.environ.get(key)
                for key in (
                    "GPU_NATIVE_EVENT_WAIT",
                    "GPU_NATIVE_EVENT_TRACE",
                    "GPU_MAX_HW_QUEUES",
                    "GPU_STREAMOPS_CP_WAIT",
                    "SGLANG_USE_AITER",
                    "SGLANG_ROCM_INDEXER_PROJECTION_SCHEDULE",
                    "SGLANG_ROCM_USE_MULTI_STREAM",
                    "SGLANG_DISABLE_HISPARSE_PREFETCH",
                    "SGLANG_RECORD_STEP_TIME",
                    "SGLANG_ROCM_FUSED_DECODE_MLA",
                    "SGLANG_OPT_USE_TOPK_V2",
                    "SGLANG_SET_CPU_AFFINITY",
                    "ROCM_QUICK_REDUCE_QUANTIZATION",
                    "SAFETENSORS_FAST_GPU",
                    "PYTHONNOUSERSITE",
                    "AITER_COMMIT",
                    "AITER_CONFIG_GEMM_BF16_FILE",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "SLURM_GPUS_ON_NODE",
                    "SLURM_JOB_ID",
                    "SLURM_STEP_ID",
                    "SLURM_CPUS_PER_TASK",
                    "HSA_ENABLE_SDMA",
                    "NATIVE_COHERENT_SHADOW",
                    "NATIVE_QK_ENVELOPE",
                    "NATIVE_CORRECTNESS",
                    "NATIVE_PROFILE",
                )
            },
        )
        if owner and receipt["invoked_indexers"] != [0, 1, 2, 6]:
            raise AssertionError(
                f"Invoked indexer census mismatch: {receipt['invoked_indexers']}"
            )
        _write(f"worker-{role}-rank{rank}.json", receipt)
        tg.solMap.update(originals)
        self._native_attestation_pid = os.getpid()

    ModelRunner.init_cuda_graphs = initialize_graphs

    if os.environ.get("NATIVE_PROFILE") == "1" and diagnostic is None:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        execute_original = DecodeCudaGraphRunner.execute
        profiled = set()
        eligible_counts = {}

        def execute_profile(self, forward_batch, *args, **kwargs):
            lengths = forward_batch.seq_lens_cpu
            if lengths is None:
                lengths = forward_batch.seq_lens
            eligible = (
                not self.model_runner.is_draft_worker and int(lengths.max()) > 2048
            )
            identity = id(self)
            if eligible:
                eligible_counts[identity] = eligible_counts.get(identity, 0) + 1
            if (
                not eligible
                or eligible_counts.get(identity, 0) < 3
                or identity in profiled
            ):
                return execute_original(self, forward_batch, *args, **kwargs)
            profiled.add(identity)
            torch.cuda.synchronize()
            print(
                f"NATIVE_REPLAY_BEGIN pid={os.getpid()} tokens={forward_batch.input_ids.numel()}",
                file=sys.stderr,
                flush=True,
            )
            sdk = ctypes.CDLL("/opt/rocm/lib/librocprofiler-sdk-roctx.so")
            sdk.roctxProfilerResume.argtypes = [ctypes.c_uint64]
            sdk.roctxProfilerResume.restype = ctypes.c_int
            sdk.roctxProfilerPause.argtypes = [ctypes.c_uint64]
            sdk.roctxProfilerPause.restype = ctypes.c_int
            sdk.roctxRangePushA.argtypes = [ctypes.c_char_p]
            sdk.roctxRangePushA.restype = ctypes.c_int
            sdk.roctxRangePop.argtypes = []
            sdk.roctxRangePop.restype = ctypes.c_int
            if sdk.roctxProfilerResume(0) != 0:
                raise AssertionError("SDK selected-region resume failed")
            sdk.roctxRangePushA(b"NATIVE_TARGET_REPLAY")
            try:
                output = execute_original(self, forward_batch, *args, **kwargs)
                torch.cuda.synchronize()
            finally:
                sdk.roctxRangePop()
                if sdk.roctxProfilerPause(0) != 0:
                    raise AssertionError("SDK selected-region pause failed")
            print(
                f"NATIVE_REPLAY_END pid={os.getpid()} key={self._replay_graph_key}",
                file=sys.stderr,
                flush=True,
            )
            _write(
                f"profile-window-rank{torch.distributed.get_rank()}.json",
                dict(
                    backend="rocprofv3-selected-regions",
                    pid=os.getpid(),
                    rank=torch.distributed.get_rank(),
                    graph_key=str(self._replay_graph_key),
                    input_ids=forward_batch.input_ids.cpu().tolist(),
                    max_seq_len=int(lengths.max()),
                    complete_target_replays=1,
                ),
            )
            return output

        DecodeCudaGraphRunner.execute = execute_profile


def maps_after_warmup(output):
    """Called outside timing, from the same container/UID as the worker group."""
    root = Path(output)
    digests = {}
    records = []
    for receipt_path in sorted(root.glob("worker-*-rank*.json")):
        receipt = json.loads(receipt_path.read_text())
        mappings = Path(f"/proc/{receipt['pid']}/maps").read_text()
        paths = sorted(
            {
                line.split()[-1]
                for line in mappings.splitlines()
                if len(line.split()) >= 6 and line.split()[-1].startswith("/")
            }
        )
        libraries = {}
        for name in paths:
            path = Path(name)
            if path.is_file() and (
                ".so" in path.name or path.suffix in (".hsaco", ".cubin")
            ):
                if name not in digests:
                    digest = hashlib.sha256()
                    with path.open("rb") as binary:
                        while chunk := binary.read(8 * 1024 * 1024):
                            digest.update(chunk)
                    digests[name] = digest.hexdigest()
                libraries[name] = digests[name]
        for stem, expected in receipt["runtime"]["mapped"].items():
            actual = {str(Path(path).resolve()) for path in paths if stem in path}
            if actual != {expected}:
                raise AssertionError(f"Worker runtime changed after warmup: {actual}")
        (root / f"maps-{receipt['role']}-rank{receipt['rank']}.txt").write_text(
            mappings
        )
        records.append(
            dict(
                role=receipt["role"],
                rank=receipt["rank"],
                pid=receipt["pid"],
                libraries=libraries,
            )
        )
    if len(records) != 16:
        raise AssertionError(f"Expected 16 target/draft receipts, got {len(records)}")
    (root / "warmup-library-manifest.json").write_text(json.dumps(records, indent=2))
