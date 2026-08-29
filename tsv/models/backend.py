"""Inference backend selection.

The plan commits to running the same ONNX models on whatever silicon the user
has, so nothing above this module names a runtime. The abstraction sits over
*runtimes* rather than ONNX Runtime execution providers, because
`onnxruntime-openvino` has no wheel for this Python: the Intel iGPU path has
to go through OpenVINO's own runtime instead of an ORT provider.

Selection is ordered, reported, and overridable. Which device actually ran
matters enough to the user - it is the difference between a night of footage
taking ten minutes or two hours - that the chosen backend is surfaced rather
than silently decided.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

# Ordered best-effort. Each entry is (runtime, device); the first that both
# imports and compiles the model wins.
#
# The order below is measured, not assumed. On the baseline machine (i5-1235U
# with Intel UHD graphics), yolo11n at 640px:
#
#     openvino:GPU      33 ms      the iGPU is worth having
#     onnxruntime:CPU   45 ms
#     openvino:CPU      53 ms
#
# OpenVINO's CPU path losing to ONNX Runtime's was the surprise - it sits last
# rather than ahead of ORT on the strength of being Intel's own runtime.
DEFAULT_PREFERENCE: tuple[tuple[str, str], ...] = (
    ("openvino", "GPU"),      # Intel iGPU - the baseline target's only accelerator
    ("onnxruntime", "DML"),   # DirectML, covers AMD/NVIDIA/Intel on Windows
    ("onnxruntime", "CUDA"),
    ("onnxruntime", "CPU"),   # always available, and faster than OpenVINO's CPU
    ("openvino", "CPU"),      # only reached if onnxruntime is missing entirely
)

# The best backend is a property of the *model*, not just the machine.
# Measured per forward pass on the baseline machine (i5-1235U, Intel UHD):
#
#     yolo11n      640px   iGPU  33 ms   CPU  45 ms   -> iGPU wins
#     SCRFD 500m   640px   iGPU  54 ms   CPU  24 ms   -> CPU wins
#     ArcFace mbf  112px   iGPU  24 ms   CPU   9 ms   -> CPU wins
#     CLIP image   224px   iGPU  82 ms   CPU  51 ms   -> CPU wins
#     CLIP text     77tok  iGPU  34 ms   CPU  36 ms   -> a wash
#
# The pattern is not model size - CLIP's image encoder is by far the largest
# graph here and still loses on the iGPU. It is that only a sustained,
# convolution-heavy workload keeps that GPU busy enough to pay back the cost
# of getting data to it; everything else is dominated by transfer and launch
# overhead. Compile time compounds it: the iGPU takes 4-6 seconds to compile a
# graph the CPU loads in half a second, which a user feels directly on their
# first search.
#
# A discrete GPU has the headroom to win in all these cases, so it stays first;
# only the integrated one is demoted.
CPU_FIRST_PREFERENCE: tuple[tuple[str, str], ...] = (
    ("onnxruntime", "CUDA"),
    ("onnxruntime", "CPU"),
    ("openvino", "GPU"),
    ("onnxruntime", "DML"),
    ("openvino", "CPU"),
)

# Kept as the old name so nothing that imported it breaks.
SMALL_MODEL_PREFERENCE = CPU_FIRST_PREFERENCE


@dataclass
class BackendInfo:
    runtime: str
    device: str
    # Why the earlier candidates were passed over, in order. Empty when the
    # first choice worked.
    rejected: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.runtime}:{self.device}"


class Backend(Protocol):
    info: BackendInfo
    input_names: list[str]
    # The first input's shape as the graph declares it, with None for any
    # dynamic axis. Worth having because a static export decides its own input
    # size - YOLOX-tiny is 416, YOLO11 is 640 - and feeding the wrong one is a
    # hard failure at inference rather than something a config default can be
    # trusted to get right.
    input_shape: tuple[int | None, ...]

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...


class OnnxRuntimeBackend:
    def __init__(self, model_path: Path, device: str, threads: int | None = None) -> None:
        import onnxruntime as ort

        provider = {
            "CPU": "CPUExecutionProvider",
            "DML": "DmlExecutionProvider",
            "CUDA": "CUDAExecutionProvider",
        }[device]
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"{provider} not available in this onnxruntime build")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            # The baseline laptop has 2 performance cores and 8 efficient ones;
            # letting ORT spawn a thread per logical core makes it slower, not
            # faster, because the pool waits on the slowest worker.
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=[provider]
        )
        inputs = self._session.get_inputs()
        self.input_names = [i.name for i in inputs]
        self.input_shape = tuple(
            d if isinstance(d, int) else None for d in (inputs[0].shape if inputs else ())
        )
        self._output_names = [o.name for o in self._session.get_outputs()]
        self.info = BackendInfo("onnxruntime", device)

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._session.run(self._output_names, inputs)


class OpenVINOBackend:
    def __init__(self, model_path: Path, device: str, threads: int | None = None) -> None:
        import openvino as ov

        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(f"OpenVINO device {device} not present")

        # LATENCY, not THROUGHPUT. The pipeline submits one letterboxed frame
        # at a time and blocks on the result; the throughput hint optimises for
        # concurrent streams instead and measured 8x slower per frame on CPU.
        config: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
        if threads and device == "CPU":
            config["INFERENCE_NUM_THREADS"] = str(threads)

        compiled = core.compile_model(str(model_path), device, config)
        self._request = compiled.create_infer_request()
        self._compiled = compiled
        self.input_names = [next(iter(p.names)) for p in compiled.inputs]
        self.input_shape = ()
        if compiled.inputs:
            shape = compiled.inputs[0].partial_shape
            self.input_shape = tuple(
                d.get_length() if d.is_static else None for d in shape
            )
        self.info = BackendInfo("openvino", device)

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        result = self._request.infer(inputs)
        return [result[output] for output in self._compiled.outputs]


def _default_threads() -> int:
    """Performance cores only, where that can be told apart from the total."""
    total = os.cpu_count() or 4
    return max(1, min(8, total // 2))


def load_model(
    model_path: Path,
    preference: tuple[tuple[str, str], ...] = DEFAULT_PREFERENCE,
    threads: int | None = None,
    force: str | None = None,
) -> Backend:
    """Compile `model_path` on the best available backend.

    `force` takes "runtime:device" (e.g. "onnxruntime:CPU") and skips
    selection entirely, which is what benchmarks and tests want.
    """
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    threads = threads or _default_threads()

    candidates = preference
    if force:
        runtime, _, device = force.partition(":")
        candidates = ((runtime, device or "CPU"),)

    rejected: list[str] = []
    for runtime, device in candidates:
        try:
            backend: Backend
            if runtime == "onnxruntime":
                backend = OnnxRuntimeBackend(model_path, device, threads)
            elif runtime == "openvino":
                backend = OpenVINOBackend(model_path, device, threads)
            else:
                raise RuntimeError(f"unknown runtime {runtime!r}")
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next one"
            rejected.append(f"{runtime}:{device} ({type(exc).__name__}: {exc})")
            continue
        backend.info.rejected = rejected
        return backend

    raise RuntimeError(
        "no usable inference backend. Tried:\n  " + "\n  ".join(rejected)
    )


def benchmark(backend: Backend, inputs: dict[str, np.ndarray], runs: int = 20) -> float:
    """Median seconds per inference, after a warmup."""
    for _ in range(3):
        backend.run(inputs)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        backend.run(inputs)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))
