"""
Particle Swarm Optimization + Surrogate-Based Optimization (PSO+SBO)
for VLSI Placement.

Combines PSO's population-based exploration with surrogate-assisted
fitness evaluation to efficiently search the placement space.

Each particle encodes cell-cluster centroid positions.
The surrogate (RBF) pre-screens particles to save expensive evaluations.

PSO update rules:
    v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x(t+1) = x(t) + v(t+1)
"""

import numpy as np
import time
from scipy.interpolate import RBFInterpolator
from typing import Optional, Callable, Dict, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from tqdm import trange

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.placement import PlacementData
from core.objectives import PlacementObjective
from core.density import compute_density_map_fast


class PSOWithSBO:
    """
    Particle Swarm Optimization with Surrogate-Based acceleration.
    """

    def __init__(
        self,
        data: PlacementData,
        objective: PlacementObjective,
        num_clusters: int = 50,
        num_particles: int = 30,
        max_iterations: int = 100,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        surrogate_update_every: int = 5,
        seed: int = 42,
        callback: Optional[Callable] = None,
        n_jobs: int = 1,
        use_numba: bool = False,
        use_gpu: bool = False,
    ):
        self.data = data
        self.objective = objective
        self.num_clusters = num_clusters
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.surrogate_update_every = surrogate_update_every
        self.rng = np.random.RandomState(seed)
        self.callback = callback
        self.n_jobs = max(1, n_jobs)
        self.use_numba = use_numba
        self.use_gpu = use_gpu

        self.cluster_labels = None
        self.base_centroids = None
        self.history: List[Dict] = []

        # Surrogate
        self.surrogate = None
        self.surrogate_X = []
        self.surrogate_Y = []

    def _init_clusters(self):
        """K-means clustering of movable cells."""
        positions = self.data.get_movable_positions()
        n = len(positions)
        k = min(self.num_clusters, n)

        indices = self.rng.choice(n, size=k, replace=False)
        centroids = positions[indices].copy()

        for _ in range(20):
            dists = np.linalg.norm(positions[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
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
        self.base_centroids = centroids.copy()
        return centroids

    def _apply_cluster_positions(self, cluster_pos: np.ndarray):
        """Move cells according to cluster centroid displacements."""
        positions = self.data.get_movable_positions()
        for j in range(len(cluster_pos)):
            mask = self.cluster_labels == j
            if not np.any(mask):
                continue
            delta = cluster_pos[j] - self.base_centroids[j]
            positions[mask] += delta

        self.data.set_movable_positions(positions)
        self.data.clip_to_die()
        self.base_centroids = cluster_pos.copy()

    def _evaluate_true(self, flat_pos: np.ndarray) -> float:
        """Evaluate with true objective function."""
        # Save current state
        saved = self.data.get_movable_positions().copy()
        saved_centroids = self.base_centroids.copy()

        self._apply_cluster_positions(flat_pos.reshape(-1, 2))
        result = self.objective.evaluate()
        score = result['total']

        # Store for surrogate training
        self.surrogate_X.append(flat_pos.copy())
        self.surrogate_Y.append(score)

        # Restore state
        self.data.set_movable_positions(saved)
        self.base_centroids = saved_centroids

        return score

    def _build_surrogate(self):
        """Build/update RBF surrogate model."""
        if len(self.surrogate_X) < 10:
            return

        X = np.array(self.surrogate_X)
        Y = np.array(self.surrogate_Y)

        self._surr_mean = X.mean(axis=0)
        self._surr_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._surr_mean) / self._surr_std

        try:
            self.surrogate = RBFInterpolator(X_norm, Y, kernel='thin_plate_spline', smoothing=1.0)
        except Exception:
            self.surrogate = None

    def _evaluate_surrogate(self, flat_pos: np.ndarray) -> float:
        """Fast surrogate evaluation."""
        if self.surrogate is None:
            return self._evaluate_true(flat_pos)
        x_norm = (flat_pos.reshape(1, -1) - self._surr_mean) / self._surr_std
        return float(self.surrogate(x_norm)[0])

    def _batch_evaluate(self, particle_positions: List[np.ndarray]) -> List[float]:
        """Batch evaluate multiple candidates (placeholder for Phase 1).
        
        In Phase 1, this uses ProcessPoolExecutor.
        In Phase 3, this will batch computations and amortize density map overhead.
        """
        if self.n_jobs <= 1 or len(particle_positions) <= 1:
            # Sequential evaluation
            return [self._evaluate_true(p) for p in particle_positions]
        
        # Parallel evaluation using ThreadPoolExecutor
        # Note: ProcessPoolExecutor not used here due to pickling challenges with large object data
        # ThreadPoolExecutor works for I/O-bound operations; compute is still NumPy-parallelized
        scores = []
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(particle_positions))) as executor:
            futures = [executor.submit(self._evaluate_true, pos) for pos in particle_positions]
            scores = [f.result() for f in futures]
        return scores

    def run(self) -> Dict:
        """Execute PSO+SBO algorithm."""
        start_time = time.time()
        print("\n" + "=" * 60)
        print("  PSO + SBO Placement Optimizer")
        print("=" * 60)

        # Initialize clusters
        print("[PSO+SBO] Initializing cell clusters...")
        centroids = self._init_clusters()
        dim = centroids.size  # 2 * num_clusters
        print(f"  -> {len(centroids)} clusters, {dim} dimensions")

        # Initialize particle swarm
        print(f"[PSO+SBO] Initializing {self.num_particles} particles...")
        particles = np.zeros((self.num_particles, dim))
        velocities = np.zeros((self.num_particles, dim))
        pbest_pos = np.zeros((self.num_particles, dim))
        pbest_score = np.full(self.num_particles, np.inf)
        gbest_pos = None
        gbest_score = np.inf

        # Position bounds
        lb = np.tile([self.data.die_xl + 50, self.data.die_yl + 50], len(centroids))
        ub = np.tile([self.data.die_xh - 50, self.data.die_yh - 50], len(centroids))
        v_max = (ub - lb) * 0.1

        # Initialize particles around base centroids
        base_flat = centroids.flatten()
        for i in range(self.num_particles):
            if i == 0:
                particles[i] = base_flat.copy()
            else:
                noise = self.rng.randn(dim) * (self.data.die_width * 0.08)
                particles[i] = np.clip(base_flat + noise, lb, ub)
            velocities[i] = self.rng.randn(dim) * (self.data.die_width * 0.01)

        # Evaluate initial particles
        print("[PSO+SBO] Evaluating initial swarm...")
        for i in range(self.num_particles):
            score = self._evaluate_true(particles[i])
            pbest_pos[i] = particles[i].copy()
            pbest_score[i] = score
            if score < gbest_score:
                gbest_score = score
                gbest_pos = particles[i].copy()

        print(f"  -> Initial gbest: {gbest_score:.4f}")

        # Main PSO loop
        print(f"[PSO+SBO] Starting optimization ({self.max_iterations} iterations)...")
        for iteration in trange(self.max_iterations, desc="PSO Optimization", unit="iter"):
            # Linearly decreasing inertia weight
            w = self.w_start - (self.w_start - self.w_end) * iteration / self.max_iterations

            # Update surrogate periodically
            if (iteration + 1) % self.surrogate_update_every == 0:
                self._build_surrogate()

            # Update each particle
            for i in range(self.num_particles):
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)

                # Velocity update
                velocities[i] = (
                    w * velocities[i] +
                    self.c1 * r1 * (pbest_pos[i] - particles[i]) +
                    self.c2 * r2 * (gbest_pos - particles[i])
                )

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Position update
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate (use surrogate for most, true for a subset)
                if self.surrogate is not None and self.rng.random() < 0.7:
                    score = self._evaluate_surrogate(particles[i])
                else:
                    score = self._evaluate_true(particles[i])

                # Update personal best
                if score < pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i] = particles[i].copy()

                # Update global best
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = particles[i].copy()

            # Record history
            record = {
                'iteration': iteration,
                'gbest': gbest_score,
                'avg_score': float(np.mean(pbest_score)),
                'inertia': w,
            }
            self.history.append(record)

            if self.callback:
                self.callback(iteration, record)

            if (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1:3d}/{self.max_iterations} | "
                      f"gBest: {gbest_score:.4f} | "
                      f"Avg: {np.mean(pbest_score):.4f} | "
                      f"w: {w:.2f} | "
                      f"Surr. samples: {len(self.surrogate_X)}")

        # Apply best solution
        print("[PSO+SBO] Applying best placement...")
        self._apply_cluster_positions(gbest_pos.reshape(-1, 2))
        final_result = self.objective.evaluate()

        runtime = time.time() - start_time
        print(f"\n[PSO+SBO] Done in {runtime:.1f}s")
        print(f"  Final HPWL:     {final_result['hpwl']:.0f}")
        print(f"  Final Overflow: {final_result['density_overflow']:.4f}")
        print("=" * 60)

        return {
            'hpwl': final_result['hpwl'],
            'density_overflow': final_result['density_overflow'],
            'density_map': final_result['density_map'],
            'runtime': runtime,
            'history': self.history,
            'algorithm': 'PSO + SBO',
        }
