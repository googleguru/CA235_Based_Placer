"""
Hybrid Surrogate-Based Optimization (SBO) for VLSI Placement.

Replaces DREAMPlace's gradient-based (PyTorch Adam/Nesterov) optimizer with
a derivative-free surrogate-based approach:

1. Sample initial placements and evaluate objective
2. Build RBF surrogate model of the objective landscape
3. Optimize the surrogate to find promising candidate positions
4. Evaluate true objective at candidates, update surrogate
5. Apply local refinement (Nelder-Mead) at the best solution
6. Repeat until convergence

The "hybrid" aspect combines global surrogate exploration with
local optimization for exploitation.
"""

import numpy as np
import time
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from typing import Optional, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import trange

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.placement import PlacementData
from core.objectives import PlacementObjective
from core.density import compute_density_map_fast, compute_target_density
from core.wirelength import compute_hpwl


class HybridSBO:
    """
    Hybrid Surrogate-Based Optimization for cell placement.

    For tractability with large designs, operates on cell *clusters*
    rather than individual cells. Cells are grouped into K clusters,
    and the algorithm optimizes cluster centroid positions.
    """

    def __init__(
        self,
        data: PlacementData,
        objective: PlacementObjective,
        num_clusters: int = 50,
        initial_samples: int = 30,
        max_iterations: int = 80,
        candidates_per_iter: int = 10,
        local_refine_every: int = 10,
        seed: int = 42,
        callback: Optional[Callable] = None,
        n_jobs: int = 1,
        use_numba: bool = False,
        use_gpu: bool = False,
    ):
        self.data = data
        self.objective = objective
        self.num_clusters = num_clusters
        self.initial_samples = initial_samples
        self.max_iterations = max_iterations
        self.candidates_per_iter = candidates_per_iter
        self.local_refine_every = local_refine_every
        self.rng = np.random.RandomState(seed)
        self.callback = callback
        self.n_jobs = max(1, n_jobs)
        self.use_numba = use_numba
        self.use_gpu = use_gpu

        # Cluster assignments and centroids
        self.cluster_labels = None
        self.cluster_centroids = None
        self.movable_idx = data.get_movable_indices()

        # Surrogate data
        self.X_samples: List[np.ndarray] = []  # position samples
        self.Y_samples: List[float] = []         # objective values
        self.surrogate = None

        # History
        self.history: List[Dict] = []
        self.best_objective = np.inf
        self.best_positions = None

    def _init_clusters(self):
        """Group movable cells into K clusters using k-means-like approach."""
        positions = self.data.get_movable_positions()
        n = len(positions)
        k = min(self.num_clusters, n)

        # Simple k-means initialization
        indices = self.rng.choice(n, size=k, replace=False)
        centroids = positions[indices].copy()

        for _ in range(20):
            # Assign to nearest centroid
            dists = np.linalg.norm(positions[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = positions[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=1e-3):
                break
            centroids = new_centroids

        self.cluster_labels = labels
        self.cluster_centroids = centroids
        return centroids

    def _apply_cluster_positions(self, cluster_pos: np.ndarray):
        """
        Move cells based on cluster centroid displacements.
        Each cell moves by the same delta as its cluster centroid.
        """
        positions = self.data.get_movable_positions()

        for j in range(len(cluster_pos)):
            mask = self.cluster_labels == j
            if not np.any(mask):
                continue
            # Shift cells by the difference between new and old centroid
            delta = cluster_pos[j] - self.cluster_centroids[j]
            positions[mask] += delta

        self.data.set_movable_positions(positions)
        self.data.clip_to_die()
        self.cluster_centroids = cluster_pos.copy()

    def _evaluate(self, cluster_pos: np.ndarray) -> float:
        """Evaluate objective given cluster positions."""
        self._apply_cluster_positions(cluster_pos.reshape(-1, 2))
        result = self.objective.evaluate()
        return result['total']

    def _sample_random(self) -> np.ndarray:
        """Generate a random cluster position sample within die bounds."""
        k = len(self.cluster_centroids)
        sample = np.zeros((k, 2))
        sample[:, 0] = self.rng.uniform(self.data.die_xl + 100, self.data.die_xh - 100, k)
        sample[:, 1] = self.rng.uniform(self.data.die_yl + 100, self.data.die_yh - 100, k)
        return sample

    def _build_surrogate(self):
        """Build/update the RBF surrogate model."""
        X = np.array(self.X_samples)
        Y = np.array(self.Y_samples)

        # Normalize inputs
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._X_mean) / self._X_std

        try:
            self.surrogate = RBFInterpolator(X_norm, Y, kernel='thin_plate_spline', smoothing=1.0)
        except Exception:
            self.surrogate = RBFInterpolator(X_norm, Y, kernel='linear', smoothing=1.0)

    def _query_surrogate(self, x: np.ndarray) -> float:
        """Query the surrogate at a point."""
        x_norm = (x.reshape(1, -1) - self._X_mean) / self._X_std
        return float(self.surrogate(x_norm)[0])

    def _optimize_surrogate(self, num_candidates: int) -> np.ndarray:
        """Find promising points by perturbing current best and evaluating surrogate."""
        best_idx = np.argmin(self.Y_samples)
        best_x = self.X_samples[best_idx].copy()

        candidates = []
        scores = []

        for _ in range(num_candidates * 5):
            # Perturb the best solution
            noise_scale = (self.data.die_width + self.data.die_height) * 0.05
            candidate = best_x + self.rng.randn(len(best_x)) * noise_scale
            # Clip to die bounds
            candidate = np.clip(candidate, 
                              [self.data.die_xl + 50, self.data.die_yl + 50] * (len(best_x) // 2),
                              [self.data.die_xh - 50, self.data.die_yh - 50] * (len(best_x) // 2))
            
            score = self._query_surrogate(candidate)
            candidates.append(candidate)
            scores.append(score)

        # Return the best candidate according to surrogate
        scores = np.array(scores)
        best_candidates = np.argsort(scores)[:num_candidates]
        return [candidates[i] for i in best_candidates]

    def run(self) -> Dict:
        """
        Execute the Hybrid SBO algorithm.

        Returns:
            dict with 'hpwl', 'density_overflow', 'runtime', 'history'
        """
        start_time = time.time()
        print("\n" + "=" * 60)
        print("  Hybrid SBO Placement Optimizer")
        print("=" * 60)

        # Step 1: Initialize clusters
        print("[SBO] Initializing cell clusters...")
        base_centroids = self._init_clusters()
        print(f"  -> {len(base_centroids)} clusters created")

        # Step 2: Initial sampling
        print(f"[SBO] Generating {self.initial_samples} initial samples...")
        for i in range(self.initial_samples):
            if i == 0:
                sample = base_centroids.copy()
            else:
                # Perturb base positions
                noise = self.rng.randn(*base_centroids.shape) * (self.data.die_width * 0.1)
                sample = base_centroids + noise
                sample[:, 0] = np.clip(sample[:, 0], self.data.die_xl + 50, self.data.die_xh - 50)
                sample[:, 1] = np.clip(sample[:, 1], self.data.die_yl + 50, self.data.die_yh - 50)

            flat = sample.flatten()
            obj = self._evaluate(flat.reshape(-1, 2))
            self.X_samples.append(flat)
            self.Y_samples.append(obj)

            if obj < self.best_objective:
                self.best_objective = obj
                self.best_positions = self.data.get_movable_positions().copy()

        print(f"  -> Initial best objective: {self.best_objective:.4f}")

        # Step 3: Main optimization loop
        print(f"[SBO] Starting optimization ({self.max_iterations} iterations)...")
        for iteration in trange(self.max_iterations, desc="SBO Optimization", unit="iter"):
            # Build surrogate
            self._build_surrogate()

            # Find promising candidates via surrogate
            candidates = self._optimize_surrogate(self.candidates_per_iter)

            # Evaluate candidates with true objective
            for candidate in candidates:
                obj = self._evaluate(candidate.reshape(-1, 2))
                self.X_samples.append(candidate)
                self.Y_samples.append(obj)

                if obj < self.best_objective:
                    self.best_objective = obj
                    self.best_positions = self.data.get_movable_positions().copy()

            # Local refinement periodically
            if (iteration + 1) % self.local_refine_every == 0:
                self._local_refine()

            # Record history
            result = self.objective.evaluate()
            record = {
                'iteration': iteration,
                'objective': self.best_objective,
                'hpwl': result['hpwl'],
                'density_overflow': result['density_overflow'],
                'num_samples': len(self.X_samples),
            }
            self.history.append(record)

            if self.callback:
                self.callback(iteration, record)

            if (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1:3d}/{self.max_iterations} | "
                      f"Obj: {self.best_objective:.4f} | "
                      f"HPWL: {result['hpwl']:.0f} | "
                      f"Overflow: {result['density_overflow']:.4f} | "
                      f"Samples: {len(self.X_samples)}")

        # Restore best placement
        if self.best_positions is not None:
            self.data.set_movable_positions(self.best_positions)
            self.data.clip_to_die()

        runtime = time.time() - start_time
        final_result = self.objective.evaluate()

        print(f"\n[SBO] Done in {runtime:.1f}s")
        print(f"  Final HPWL:     {final_result['hpwl']:.0f}")
        print(f"  Final Overflow: {final_result['density_overflow']:.4f}")
        print("=" * 60)

        return {
            'hpwl': final_result['hpwl'],
            'density_overflow': final_result['density_overflow'],
            'density_map': final_result['density_map'],
            'runtime': runtime,
            'history': self.history,
            'algorithm': 'Hybrid SBO',
        }

    def _local_refine(self):
        """Apply Nelder-Mead local refinement around the best solution."""
        best_idx = np.argmin(self.Y_samples)
        x0 = self.X_samples[best_idx].copy()

        def cost(flat):
            return self._evaluate(flat.reshape(-1, 2))

        try:
            result = minimize(cost, x0, method='Nelder-Mead',
                            options={'maxiter': 50, 'xatol': 10, 'fatol': 0.01})
            if result.fun < self.best_objective:
                self.best_objective = result.fun
                self.best_positions = self.data.get_movable_positions().copy()
                self.X_samples.append(result.x)
                self.Y_samples.append(result.fun)
        except Exception:
            pass  # Local refinement failed, continue with global search
