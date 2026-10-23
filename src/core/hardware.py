"""Phase 5 — hardware-adaptive compute profile for the offline engine.

`ComputeTier` selects a deterministic compute budget (DuckDB capacity, graph
sampling, IsolationForest capacity and diffusion depth) so the offline pipeline
can run lighter on a laptop and deeper on workstation hardware without changing
the science. Every knob is a pure function of the tier; callers may override the
auto-detected tier explicitly (for tests, demos and constrained machines).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Final, Literal

ComputeTier = Literal["LITE", "STANDARD", "WORKSTATION"]

# (duckdb_threads, duckdb_memory_limit, betweenness_sample_k, max_cycle_length,
#  iforest_estimators, enable_knn_density, diffusion_iterations)
_LITE_KNOBS: Final[tuple[int, str, int | None, int, int, bool, int]] = (
    2, "256MB", 64, 3, 100, False, 5,
)
_STANDARD_KNOBS: Final[tuple[int, str, int | None, int, int, bool, int]] = (
    4, "1GB", 256, 4, 200, True, 10,
)
_WORKSTATION_KNOBS: Final[tuple[int, str, int | None, int, int, bool, int]] = (
    8, "4GB", None, 5, 400, True, 20,
)


@dataclass(frozen=True)
class HardwareProfile:
    """Deterministic compute budget derived from a :data:`ComputeTier`."""

    tier: ComputeTier
    cpu_cores: int
    total_ram_gb: float
    gpu_detected: bool
    duckdb_threads: int
    duckdb_memory_limit: str
    betweenness_sample_k: int | None
    max_cycle_length: int
    iforest_estimators: int
    enable_knn_density: bool
    diffusion_iterations: int


def _knobs_for(tier: ComputeTier) -> tuple[int, str, int | None, int, int, bool, int]:
    if tier == "LITE":
        return _LITE_KNOBS
    if tier == "STANDARD":
        return _STANDARD_KNOBS
    return _WORKSTATION_KNOBS


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _gpu_detected() -> bool:
    """True when an NVIDIA GPU is visible (CUDA env var or local driver)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return True
    return shutil.which("nvidia-smi") is not None


def _fallback_system_memory_gb() -> float:
    """Best-effort platform probe with a safe default (8 GiB)."""
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
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

            windll = getattr(ctypes, "windll", None)
            if windll is None:
                return 8.0
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(float(status.ullTotalPhys) / (1024.0 * 1024.0), 2)
        except (OSError, AttributeError, ValueError):
            return 8.0
    return 8.0


def _system_memory_gb() -> float:
    """Total system RAM in GiB (Linux `/proc/meminfo`, safe fallback)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
    except (OSError, ValueError, IndexError):
        pass
    return _fallback_system_memory_gb()


def _auto_tier() -> ComputeTier:
    if _gpu_detected() and _system_memory_gb() >= 16.0 and _cpu_count() >= 16:
        return "WORKSTATION"
    if _system_memory_gb() >= 8.0 and _cpu_count() >= 8:
        return "STANDARD"
    return "LITE"


def detect_hardware_profile(override_tier: ComputeTier | None = None) -> HardwareProfile:
    """Resolve the compute budget for a tier (explicit override or auto-detect)."""
    tier: ComputeTier = override_tier if override_tier is not None else _auto_tier()
    duckdb_threads, duckdb_memory_limit, betweenness_sample_k, max_cycle_length, iforest_estimators, enable_knn_density, diffusion_iterations = _knobs_for(tier)
    return HardwareProfile(
        tier=tier,
        cpu_cores=_cpu_count(),
        total_ram_gb=_system_memory_gb(),
        gpu_detected=_gpu_detected(),
        duckdb_threads=duckdb_threads,
        duckdb_memory_limit=duckdb_memory_limit,
        betweenness_sample_k=betweenness_sample_k,
        max_cycle_length=max_cycle_length,
        iforest_estimators=iforest_estimators,
        enable_knn_density=enable_knn_density,
        diffusion_iterations=diffusion_iterations,
    )


__all__ = [
    "ComputeTier",
    "HardwareProfile",
    "detect_hardware_profile",
]
