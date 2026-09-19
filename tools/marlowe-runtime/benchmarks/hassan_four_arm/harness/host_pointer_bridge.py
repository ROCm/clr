"""Startup-only equivalent of the pinned missing host-pointer accessor.

No GPU kernels or installed package files are replaced. The HIP API is resolved
from the source-locked overlay. The bridge is recorded in every worker receipt.
"""

import ctypes
import hashlib
import os
from pathlib import Path

import torch

_state = {}


def get_device_accessible_ptr(tensor, device_index):
    if device_index < 0:
        raise RuntimeError("Target device index must be non-negative")
    target = torch.device("cuda", device_index)
    with torch.cuda.device(target):
        if tensor.is_cuda:
            if tensor.device != target:
                raise RuntimeError("GPU tensor must be on the target device")
            return tensor.data_ptr()
        if tensor.device.type != "cpu":
            raise RuntimeError("Only CPU and target-device tensors are supported")
        device_pointer = ctypes.c_void_p()
        result = _state["runtime"].hipHostGetDevicePointer(
            ctypes.byref(device_pointer), ctypes.c_void_p(tensor.data_ptr()), 0
        )
        if result != 0:
            message = _state["runtime"].hipGetErrorString(result).decode()
            raise RuntimeError(f"hipHostGetDevicePointer: {message} ({result})")
        return device_pointer.value or 0


def install():
    import sgl_kernel.kvcacheio as kvcacheio

    has_python = hasattr(kvcacheio, "get_device_accessible_ptr")
    has_op = hasattr(torch.ops.sgl_kernel, "get_device_accessible_ptr")
    if has_python or has_op:
        raise AssertionError(
            f"Unexpected bridge baseline: python={has_python}, op={has_op}"
        )
    runtime = Path(os.environ["NATIVE_RUNTIME_RELEASE"]) / "lib/libamdhip64.so"
    _state["runtime"] = ctypes.CDLL(str(runtime.resolve()))
    _state["runtime"].hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _state["runtime"].hipHostGetDevicePointer.restype = ctypes.c_int
    _state["runtime"].hipGetErrorString.argtypes = [ctypes.c_int]
    _state["runtime"].hipGetErrorString.restype = ctypes.c_char_p
    library = torch.library.Library("sgl_kernel", "FRAGMENT")
    library.define("get_device_accessible_ptr(Tensor tensor, int device_index) -> int")
    library.impl(
        "get_device_accessible_ptr",
        get_device_accessible_ptr,
        "CompositeExplicitAutograd",
    )
    _state["library"] = library
    kvcacheio.get_device_accessible_ptr = (
        torch.ops.sgl_kernel.get_device_accessible_ptr.default
    )
    return {
        "purpose": "startup host-pointer accessor compatibility only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hip_library": str(runtime.resolve()),
        "pinned_semantics": "fb91baedab3e1de668d6e8f391ccd81cab9e9acd:python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu:20-50",
    }
