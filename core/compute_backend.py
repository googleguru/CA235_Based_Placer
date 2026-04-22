"""
Compute Backend Abstraction Layer for DREAMPlace_MetaOpt

Supports multiple backend implementations:
1. NumPy (CPU, baseline)
2. Numba (CPU, JIT-compiled, 2-3× faster)
3. CuPy (GPU, 10-20× faster) [Phase 2b]

This module abstracts the compute backend so algorithms can switch
between implementations transparently based on availability and user flags.

Usage in algorithms:
    from core import compute_backend
    backend = compute_backend.get_backend(use_numba=True, use_gpu=False)
    density = backend.compute_density_map_fast(data)
    hpwl = backend.compute_hpwl(data)
"""

import numpy as np
import sys
import os
from typing import Optional, Tuple

# Import base NumPy implementations
from .density import (
    compute_density_map_fast as _numpy_density_fast,
    compute_target_density,
    density_overflow,
    density_penalty
)
from .wirelength import compute_hpwl as _numpy_hpwl
from .placement import PlacementData

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    jit = None
    prange = range

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


# ============================================================================
# Numba JIT-Compiled Functions (Phase 2a)
# ============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def _numba_density_fast_kernel(
        cell_x: np.ndarray,
        cell_y: np.ndarray,
        cell_w: np.ndarray,
        cell_h: np.ndarray,
        die_xl: float,
        die_yl: float,
        bin_width: float,
        bin_height: float,
        nbx: int,
        nby: int,
        bin_area: float,
    ) -> np.ndarray:
        """JIT-compiled density map kernel using Numba (non-parallel version for robustness)."""
        density = np.zeros((nby, nbx), dtype=np.float64)

        # Cell centers
        center_x = cell_x + cell_w / 2.0
        center_y = cell_y + cell_h / 2.0

        # Cell areas
        cell_area = cell_w * cell_h

        # Compute densities with direct loop
        for i in range(len(cell_x)):
            bi_x = int((center_x[i] - die_xl) / bin_width)
            bi_y = int((center_y[i] - die_yl) / bin_height)

            # Manual clip bounds check
            if bi_x < 0:
                bi_x = 0
            elif bi_x >= nbx:
                bi_x = nbx - 1

            if bi_y < 0:
                bi_y = 0
            elif bi_y >= nby:
                bi_y = nby - 1

            density[bi_y, bi_x] += cell_area[i] / bin_area

        return density


    @jit(nopython=True, parallel=True, cache=True)
    def _numba_hpwl_kernel(
        net_list: np.ndarray,
        net_starts: np.ndarray,
        cell_x: np.ndarray,
        cell_y: np.ndarray,
    ) -> float:
        """JIT-compiled HPWL computation using Numba."""
        total_hpwl = 0.0

        for net_idx in prange(len(net_starts) - 1):
            start = net_starts[net_idx]
            end = net_starts[net_idx + 1]

            if start >= end:
                continue

            # Find bounding box of pins in this net
            min_x = cell_x[net_list[start]]
            max_x = min_x
            min_y = cell_y[net_list[start]]
            max_y = min_y

            for pin_idx in range(start + 1, end):
                cell_idx = net_list[pin_idx]
                x = cell_x[cell_idx]
                y = cell_y[cell_idx]

                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

            # Half-perimeter wirelength
            hpwl = (max_x - min_x) + (max_y - min_y)
            total_hpwl += hpwl

        return total_hpwl


# ============================================================================
# Backend Interface Classes
# ============================================================================

class ComputeBackend:
    """Abstract base class for compute backends."""

    def __init__(self, name: str):
        self.name = name

    def compute_density_map_fast(self, data: PlacementData) -> np.ndarray:
        """Compute cell density map (fast, center-based binning)."""
        raise NotImplementedError

    def compute_hpwl(self, data: PlacementData) -> float:
        """Compute half-perimeter wirelength."""
        raise NotImplementedError


class NumpyBackend(ComputeBackend):
    """Pure NumPy backend (CPU, baseline)."""

    def __init__(self):
        super().__init__("numpy")

    def compute_density_map_fast(self, data: PlacementData) -> np.ndarray:
        return _numpy_density_fast(data)

    def compute_hpwl(self, data: PlacementData) -> float:
        return _numpy_hpwl(data)


class NumbaBackend(ComputeBackend):
    """Numba JIT-compiled backend (CPU, 2-3× faster)."""

    def __init__(self):
        if not HAS_NUMBA:
            raise RuntimeError(
                "Numba not available. Install with: pip install numba"
            )
        super().__init__("numba")

    def compute_density_map_fast(self, data: PlacementData) -> np.ndarray:
        """Fast density computation using Numba JIT."""
        bin_area = data.bin_width * data.bin_height
        if bin_area == 0:
            return np.zeros((data.num_bins_y, data.num_bins_x), dtype=np.float64)

        return _numba_density_fast_kernel(
            cell_x=data.cell_x,
            cell_y=data.cell_y,
            cell_w=data.cell_w,
            cell_h=data.cell_h,
            die_xl=data.die_xl,
            die_yl=data.die_yl,
            bin_width=data.bin_width,
            bin_height=data.bin_height,
            nbx=data.num_bins_x,
            nby=data.num_bins_y,
            bin_area=bin_area,
        )

    def compute_hpwl(self, data: PlacementData) -> float:
        """HPWL computation using Numba JIT."""
        if not hasattr(data, '_net_list') or not hasattr(data, '_net_starts'):
            # Fallback to NumPy if net data not properly structured
            return _numpy_hpwl(data)

        return _numba_hpwl_kernel(
            net_list=data._net_list,
            net_starts=data._net_starts,
            cell_x=data.cell_x,
            cell_y=data.cell_y,
        )


class CuPyBackend(ComputeBackend):
    """CuPy GPU backend (GPU, 10-20× faster) [Phase 2b, requires NVIDIA GPU]."""

    def __init__(self):
        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy not available or NVIDIA GPU not detected. "
                "Install with: pip install cupy-cuda12x (matching your CUDA version)"
            )
        super().__init__("cupy")

    def compute_density_map_fast(self, data: PlacementData) -> np.ndarray:
        """Fast GPU-accelerated density computation."""
        nbx = data.num_bins_x
        nby = data.num_bins_y
        bw = data.bin_width
        bh = data.bin_height
        bin_area = bw * bh

        if bin_area == 0:
            return np.zeros((nby, nbx), dtype=np.float64)

        # Transfer data to GPU
        cell_x_gpu = cp.asarray(data.cell_x, dtype=cp.float64)
        cell_y_gpu = cp.asarray(data.cell_y, dtype=cp.float64)
        cell_w_gpu = cp.asarray(data.cell_w, dtype=cp.float64)
        cell_h_gpu = cp.asarray(data.cell_h, dtype=cp.float64)

        # Compute on GPU
        center_x_gpu = cell_x_gpu + cell_w_gpu / 2.0
        center_y_gpu = cell_y_gpu + cell_h_gpu / 2.0

        bi_x_gpu = ((center_x_gpu - data.die_xl) / bw).astype(cp.int32)
        bi_y_gpu = ((center_y_gpu - data.die_yl) / bh).astype(cp.int32)

        # Clip bounds
        bi_x_gpu = cp.clip(bi_x_gpu, 0, nbx - 1)
        bi_y_gpu = cp.clip(bi_y_gpu, 0, nby - 1)

        cell_area_gpu = cell_w_gpu * cell_h_gpu
        density_gpu = cp.zeros((nby, nbx), dtype=cp.float64)

        # GPU histogram (much faster than CPU binning)
        cp.add.at(density_gpu, (bi_y_gpu, bi_x_gpu), cell_area_gpu / bin_area)

        # Transfer result back to CPU
        return cp.asnumpy(density_gpu)

    def compute_hpwl(self, data: PlacementData) -> float:
        """GPU-accelerated HPWL computation."""
        if not hasattr(data, '_net_list') or not hasattr(data, '_net_starts'):
            return _numpy_hpwl(data)

        # Transfer net data to GPU
        net_list_gpu = cp.asarray(data._net_list, dtype=cp.int32)
        net_starts_gpu = cp.asarray(data._net_starts, dtype=cp.int32)
        cell_x_gpu = cp.asarray(data.cell_x, dtype=cp.float64)
        cell_y_gpu = cp.asarray(data.cell_y, dtype=cp.float64)

        total_hpwl = 0.0

        for net_idx in range(len(data._net_starts) - 1):
            start = int(net_starts_gpu[net_idx])
            end = int(net_starts_gpu[net_idx + 1])

            if start >= end:
                continue

            pins_gpu = net_list_gpu[start:end]
            x_gpu = cell_x_gpu[pins_gpu]
            y_gpu = cell_y_gpu[pins_gpu]

            min_x = float(cp.min(x_gpu))
            max_x = float(cp.max(x_gpu))
            min_y = float(cp.min(y_gpu))
            max_y = float(cp.max(y_gpu))

            hpwl = (max_x - min_x) + (max_y - min_y)
            total_hpwl += hpwl

        return total_hpwl


# ============================================================================
# Backend Factory
# ============================================================================

_BACKENDS = {
    "numpy": NumpyBackend,
    "numba": NumbaBackend if HAS_NUMBA else None,
    "cupy": CuPyBackend if HAS_CUPY else None,
}


def get_backend(use_numba: bool = False, use_gpu: bool = False) -> ComputeBackend:
    """
    Get a compute backend based on user preferences and availability.

    Args:
        use_numba: Prefer Numba JIT (Phase 2a)
        use_gpu: Prefer CuPy GPU (Phase 2b, requires GPU)

    Returns:
        ComputeBackend instance (numpy, numba, or cupy)

    Raises:
        RuntimeError: If requested backend not available
    """
    if use_gpu:
        if _BACKENDS["cupy"] is not None:
            return CuPyBackend()
        else:
            print("[ComputeBackend] CuPy GPU not available, falling back to Numba")
            use_numba = True

    if use_numba:
        if _BACKENDS["numba"] is not None:
            return NumbaBackend()
        else:
            print("[ComputeBackend] Numba not available, falling back to NumPy")

    return NumpyBackend()


def list_available_backends() -> dict:
    """List all available backends and their status."""
    return {
        "numpy": "always available",
        "numba": "available" if HAS_NUMBA else "not installed (pip install numba)",
        "cupy": "available" if HAS_CUPY else "not installed (pip install cupy-cuda12x)",
    }
