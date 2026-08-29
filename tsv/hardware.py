"""What this machine can actually run.

The point of asking is to stop asking. "Do you have a good PC?" is a question
users answer wrongly in both directions - the person with a gaming GPU picks
the small model out of caution, the person on a work laptop picks the large
one and waits an hour per video. The machine can be measured instead.

Two rules, both learned the hard way in models/backend.py:

**Listed is not usable.** `onnxruntime.get_available_providers()` reports what
the wheel was compiled with, not what this machine can run. A CUDA provider
with no driver is listed and fails at session creation. So a provider is only
reported as usable here once a real graph has compiled on it.

**Bigger is not faster.** On the baseline machine the integrated GPU lost to
the CPU on every model except the detector. Capability is what this module
reports; which model is actually quicker is a measurement, and belongs to the
benchmark rather than to a specification sheet.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Below this there is no point offering the larger tiers: the models fit, but
# decoding video alongside them does not, and the machine swaps instead of
# working. Measured against the tier table in catalogue.py rather than guessed.
LOW_RAM_MB = 8 * 1024


@dataclass(frozen=True)
class Accelerator:
    """One way of running a graph on something other than the CPU."""

    runtime: str                 # "onnxruntime" | "openvino"
    device: str                  # "CUDA" | "DML" | "GPU"
    name: str = ""               # the adapter's own name, where it gives one
    vram_mb: int | None = None
    # True only once a graph has actually compiled on it. None means nothing
    # was available to try, which is not the same as a failure.
    usable: bool | None = None
    note: str = ""

    @property
    def kind(self) -> str:
        """Discrete, integrated, or unknown.

        Worth distinguishing because it is the single biggest predictor of
        whether a larger model is a good idea, and the integrated case is the
        one where the answer is usually no.
        """
        lowered = self.name.lower()
        if any(word in lowered for word in ("uhd", "iris", "vega", "radeon graphics")):
            return "integrated"
        if self.runtime == "openvino" and self.device == "GPU" and not self.name:
            return "integrated"     # OpenVINO's GPU is the Intel one
        if self.device == "CUDA" or "geforce" in lowered or "radeon rx" in lowered:
            return "discrete"
        return "unknown"

    def __str__(self) -> str:
        bits = [f"{self.runtime}:{self.device}"]
        if self.name:
            bits.append(self.name)
        if self.vram_mb:
            bits.append(f"{self.vram_mb / 1024:.1f} GB")
        return " ".join(bits)


@dataclass(frozen=True)
class Machine:
    cpu_cores: int
    ram_mb: int
    accelerators: tuple[Accelerator, ...] = ()
    os_name: str = ""

    @property
    def usable_accelerators(self) -> tuple[Accelerator, ...]:
        """Only the ones a graph has actually compiled on.

        `usable is None` - listed but never tried - is deliberately excluded.
        An untested provider is not evidence of anything, and treating it as
        capable is how a machine gets offered a model it cannot run.
        """
        return tuple(a for a in self.accelerators if a.usable is True)

    @property
    def untested_accelerators(self) -> tuple[Accelerator, ...]:
        return tuple(a for a in self.accelerators if a.usable is None)

    @property
    def best(self) -> Accelerator | None:
        """The most promising accelerator, or None for a CPU-only machine.

        Discrete before integrated, because on an integrated GPU most of these
        graphs are slower than the CPU - the reason CPU_FIRST_PREFERENCE
        exists at all.
        """
        ranked = sorted(
            self.usable_accelerators,
            key=lambda a: (
                {"discrete": 0, "unknown": 1, "integrated": 2}[a.kind],
                -(a.vram_mb or 0),
            ),
        )
        return ranked[0] if ranked else None

    @property
    def has_discrete_gpu(self) -> bool:
        return any(a.kind == "discrete" for a in self.usable_accelerators)

    @property
    def is_low_memory(self) -> bool:
        return self.ram_mb < LOW_RAM_MB

    def summary(self) -> str:
        """One line a person can read, for the setup output and the app."""
        gb = self.ram_mb / 1024
        parts = [f"{self.cpu_cores} CPU cores", f"{gb:.0f} GB RAM"]
        best = self.best
        if best:
            parts.append(str(best))
        elif self.untested_accelerators:
            # Saying "no GPU" here would be a claim nothing has checked.
            names = ", ".join(str(a) for a in self.untested_accelerators)
            parts.append(f"{names} (present, not yet tested)")
        else:
            parts.append("no GPU")
        return ", ".join(parts)


# ---------- memory ----------

def _windows_ram_mb() -> int:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return int(status.ullTotalPhys // (1024 * 1024))


def total_ram_mb() -> int:
    """Physical memory, or 0 when it cannot be determined.

    Deliberately dependency-free. Adding psutil to the *runtime* environment
    to read one number would undo part of what keeps this shippable.
    """
    try:
        if sys.platform == "win32":
            return _windows_ram_mb()
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return int(
                os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
            )
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            return int(out.stdout.strip()) // (1024 * 1024)
    except Exception:      # noqa: BLE001 - a probe must never be the thing that fails
        return 0
    return 0


# ---------- accelerators ----------

def _nvidia_adapters() -> list[tuple[str, int]]:
    """(name, VRAM MB) per NVIDIA GPU, via the driver's own tool.

    nvidia-smi ships with the driver, so its absence is itself the answer.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    found = []
    for line in out.stdout.strip().splitlines():
        name, _, memory = line.partition(",")
        try:
            found.append((name.strip(), int(memory.strip())))
        except ValueError:
            continue
    return found


def _openvino_devices() -> list[str]:
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:      # noqa: BLE001
        return []


def _onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:      # noqa: BLE001
        return []


def _verify(runtime: str, device: str, model_path: Path) -> tuple[bool, str]:
    """Compile a real graph, because being listed proves nothing."""
    from tsv.models.backend import load_model

    try:
        backend = load_model(model_path, preference=((runtime, device),))
    except Exception as exc:      # noqa: BLE001 - any failure means unusable
        return False, f"{type(exc).__name__}: {exc}"
    got = (backend.info.runtime, backend.info.device)
    if got != (runtime, device):
        return False, f"fell back to {backend.info}"
    return True, ""


def probe(verify_with: Path | None = None) -> Machine:
    """Describe this machine.

    `verify_with` is an ONNX model to compile on each candidate. Without one
    every accelerator comes back with `usable=None` - listed, untested - which
    is honest rather than optimistic. The detector is the natural choice: it
    is the smallest model and the one required for anything to work at all.
    """
    nvidia = _nvidia_adapters()
    ov_devices = _openvino_devices()
    providers = _onnxruntime_providers()

    candidates: list[Accelerator] = []
    if "CUDAExecutionProvider" in providers:
        if nvidia:
            candidates += [
                Accelerator("onnxruntime", "CUDA", name=name, vram_mb=vram)
                for name, vram in nvidia
            ]
        else:
            candidates.append(Accelerator("onnxruntime", "CUDA"))
    if "DmlExecutionProvider" in providers:
        candidates.append(
            Accelerator("onnxruntime", "DML",
                        name=nvidia[0][0] if nvidia else "",
                        vram_mb=nvidia[0][1] if nvidia else None)
        )
    if any(d.startswith("GPU") for d in ov_devices):
        candidates.append(Accelerator("openvino", "GPU"))

    checked: list[Accelerator] = []
    for candidate in candidates:
        if verify_with is None or not Path(verify_with).is_file():
            checked.append(candidate)
            continue
        ok, note = _verify(candidate.runtime, candidate.device, Path(verify_with))
        checked.append(
            Accelerator(candidate.runtime, candidate.device, candidate.name,
                        candidate.vram_mb, usable=ok, note=note)
        )

    return Machine(
        cpu_cores=os.cpu_count() or 1,
        ram_mb=total_ram_mb(),
        accelerators=tuple(checked),
        os_name=f"{platform.system()} {platform.release()}".strip(),
    )
