"""
Combined objective function for placement optimization.
Objective = α × HPWL + β × DensityPenalty

This is the function that algorithms optimize.
In DREAMPlace, this is the "loss function" passed to PyTorch optimizers.
Here, we compute it directly with NumPy (or accelerated backends: Numba, CuPy).
"""

import numpy as np
from .placement import PlacementData
from .density import compute_target_density, density_penalty
from .wirelength import compute_hpwl

# Try to import compute_backend; if not available, use fallback
try:
    from . import compute_backend as cb
    HAS_COMPUTE_BACKEND = True
except ImportError:
    HAS_COMPUTE_BACKEND = False
    # Fallback imports
    from .density import compute_density_map_fast
    from .wirelength import compute_hpwl as _fallback_hpwl


class PlacementObjective:
    """
    Evaluates placement quality combining wirelength and density.
    Supports multiple compute backends (NumPy, Numba, CuPy).
    """

    def __init__(self, data: PlacementData,
                 alpha: float = 1.0,
                 beta: float = 500.0,
                 target_util: float = 1.0,
                 use_numba: bool = False,
                 use_gpu: bool = False):
        """
        Args:
            data: Placement data
            alpha: Weight for wirelength
            beta: Weight for density penalty
            target_util: Target bin utilization (1.0 = 100%)
            use_numba: Use Numba JIT backend (Phase 2a)
            use_gpu: Use CuPy GPU backend (Phase 2b, requires GPU)
        """
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.target_density = compute_target_density(data, target_util)
        self.use_numba = use_numba
        self.use_gpu = use_gpu

        # Select compute backend
        if HAS_COMPUTE_BACKEND:
            self.backend = cb.get_backend(use_numba=use_numba, use_gpu=use_gpu)
        else:
            # Fallback to NumPy
            class FallbackBackend:
                def compute_density_map_fast(self, data):
                    return compute_density_map_fast(data)
                def compute_hpwl(self, data):
                    return _fallback_hpwl(data)
            self.backend = FallbackBackend()

        # Normalization constants (set after first evaluation)
        self._wl_norm = None
        self._dp_norm = None

        if use_numba or use_gpu:
            print(f"[PlacementObjective] Using backend: {getattr(self.backend, 'name', 'default')}")

    def evaluate(self, positions: np.ndarray = None) -> dict:
        """
        Evaluate the placement objective using selected backend.

        Args:
            positions: Optional (N_movable, 2) array. If given, sets positions first.

        Returns:
            dict with keys: 'total', 'hpwl', 'density_penalty',
                            'density_overflow', 'density_map'
        """
        if positions is not None:
            self.data.set_movable_positions(positions)
            self.data.clip_to_die()

        # Use backend for compute-intensive operations
        hpwl = self.backend.compute_hpwl(self.data)
        density_map = self.backend.compute_density_map_fast(self.data)
        dp = density_penalty(density_map, self.target_density)

        # Normalize on first call
        if self._wl_norm is None:
            self._wl_norm = max(hpwl, 1e-6)
            self._dp_norm = max(dp, 1e-6)

        wl_normalized = hpwl / self._wl_norm
        dp_normalized = dp / self._dp_norm

        total = self.alpha * wl_normalized + self.beta * dp_normalized

        overflow = float(np.sum(np.maximum(0, density_map - self.target_density)))

        return {
            'total': total,
            'hpwl': hpwl,
            'density_penalty': dp,
            'density_overflow': overflow,
            'density_map': density_map,
        }

    def evaluate_scalar(self, positions: np.ndarray = None) -> float:
        """Return just the scalar objective value."""
        return self.evaluate(positions)['total']

    def evaluate_from_flat(self, flat_pos: np.ndarray) -> float:
        """
        Evaluate from a 1D flattened position vector.
        flat_pos has shape (2*N_movable,) where [x0, y0, x1, y1, ...].
        """
        n = len(flat_pos) // 2
        positions = flat_pos.reshape(n, 2)
        return self.evaluate_scalar(positions)
