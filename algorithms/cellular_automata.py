"""
Cellular Automata CA235 for VLSI Placement.

Uses a 2D cellular automaton to iteratively spread cells from
high-density regions to low-density regions.

CA235 Rule (2D totalistic):
    - Each bin has a density state
    - Neighborhood: Moore (8 neighbors + self = 9 cells)
    - The totalistic sum determines the action:
      * If sum > threshold → spread cells outward (reduce density)
      * If sum < threshold → attract cells inward (fill voids)
      * Rule index "235" encodes specific transition thresholds

The CA naturally handles density legalization and creates smooth,
physically plausible cell distributions.

After CA-based spreading, a wirelength-aware refinement step
adjusts positions to improve HPWL.
"""

import numpy as np
import time
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


class CellularAutomataCA235:
    """
    Cellular Automata-based cell spreader and optimizer.

    The density grid is treated as a 2D CA. Each iteration:
    1. Compute density map
    2. Apply CA rule to determine cell movement directions
    3. Move cells according to the CA-generated flow field
    4. Optionally apply wirelength-driven perturbation
    """

    def __init__(
        self,
        data: PlacementData,
        objective: PlacementObjective,
        max_iterations: int = 150,
        ca_iterations_per_step: int = 3,
        move_scale: float = 0.5,
        wl_refine_weight: float = 0.3,
        density_threshold: float = 1.0,
        seed: int = 42,
        callback: Optional[Callable] = None,
        n_jobs: int = 1,
        use_numba: bool = False,
        use_gpu: bool = False,
    ):
        self.data = data
        self.objective = objective
        self.max_iterations = max_iterations
        self.ca_iterations_per_step = ca_iterations_per_step
        self.move_scale = move_scale
        self.wl_refine_weight = wl_refine_weight
        self.density_threshold = density_threshold
        self.rng = np.random.RandomState(seed)
        self.callback = callback
        self.n_jobs = max(1, n_jobs)
        self.use_numba = use_numba
        self.use_gpu = use_gpu

        self.target_density = compute_target_density(data, 1.0)
        self.history: List[Dict] = []

    def _ca_rule_235(self, density: np.ndarray) -> tuple:
        """
        Apply CA Rule 235 to the density grid.

        Rule 235 (binary: 11101011):
        Totalistic rule over Moore neighborhood (9 cells including center).
        For each bin, compute the sum of all 9 neighbor densities.
        The rule determines whether to push cells outward or inward.

        Returns:
            (flow_x, flow_y): 2D arrays indicating movement direction
                              positive = move right/up, negative = move left/down
        """
        Ny, Nx = density.shape
        flow_x = np.zeros_like(density)
        flow_y = np.zeros_like(density)

        # Pad density for boundary handling
        padded = np.pad(density, 1, mode='edge')

        # Rule 235 threshold table
        # In binary: 235 = 11101011 → bits at positions 0,1,3,5,6,7 are set
        # For sum count 0..8 (9 cells): set/unset determines spread or attract
        rule_bits = format(235, '09b')[::-1]  # LSB first, 9 bits

        for y in range(Ny):
            for x in range(Nx):
                # Moore neighborhood sum (quantized to integer based on threshold)
                neighborhood = padded[y:y+3, x:x+3]
                over_threshold = np.sum(neighborhood > self.density_threshold * self.target_density.mean())
                over_threshold = min(int(over_threshold), 8)

                # Apply rule
                if int(rule_bits[over_threshold]) == 1:
                    # SPREAD: push cells away from high-density center
                    center_density = density[y, x]
                    if center_density > self.target_density[y, x]:
                        # Compute gradient direction (away from high density)
                        dx = 0.0
                        dy = 0.0
                        if x > 0:
                            dx -= density[y, x-1]
                        if x < Nx - 1:
                            dx += density[y, x+1]
                        if y > 0:
                            dy -= density[y-1, x]
                        if y < Ny - 1:
                            dy += density[y+1, x]

                        # Normalize
                        mag = np.sqrt(dx**2 + dy**2) + 1e-10
                        excess = center_density - self.target_density[y, x]
                        flow_x[y, x] = -dx / mag * excess
                        flow_y[y, x] = -dy / mag * excess
                else:
                    # ATTRACT: slightly pull toward high-density neighbors (fill voids)
                    if density[y, x] < self.target_density[y, x] * 0.5:
                        # Find direction toward nearest high-density bin
                        dx = 0.0
                        dy = 0.0
                        if x > 0:
                            dx += density[y, x-1]
                        if x < Nx - 1:
                            dx -= density[y, x+1]
                        if y > 0:
                            dy += density[y-1, x]
                        if y < Ny - 1:
                            dy -= density[y+1, x]

                        mag = np.sqrt(dx**2 + dy**2) + 1e-10
                        deficit = self.target_density[y, x] - density[y, x]
                        flow_x[y, x] = dx / mag * deficit * 0.3
                        flow_y[y, x] = dy / mag * deficit * 0.3

        return flow_x, flow_y

    def _ca_rule_235_fast(self, density: np.ndarray) -> tuple:
        """
        Vectorized (fast) version of CA Rule 235.
        Uses convolution instead of per-cell loops.
        """
        Ny, Nx = density.shape

        # Compute excess density
        excess = density - self.target_density

        # Gradient of density (indicates direction from high to low)
        # Using numpy gradient (central differences)
        grad_y, grad_x = np.gradient(density)

        # Moore neighborhood count of high-density cells
        padded = np.pad((density > self.density_threshold * self.target_density.mean()).astype(float),
                       1, mode='constant')

        # Sum over 3x3 neighborhood
        neighbor_sum = np.zeros((Ny, Nx))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                neighbor_sum += padded[1+dy:Ny+1+dy, 1+dx:Nx+1+dx]

        # Rule 235 lookup
        rule_table = np.array([int(b) for b in format(235, '09b')[::-1]], dtype=float)
        neighbor_sum_int = np.clip(neighbor_sum.astype(int), 0, 8)
        rule_output = rule_table[neighbor_sum_int]

        # Flow field
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10

        # Where rule says SPREAD (1): move away from gradient (reduce density)
        spread_mask = (rule_output == 1) & (excess > 0)
        attract_mask = (rule_output == 0) & (excess < 0)

        flow_x = np.zeros_like(density)
        flow_y = np.zeros_like(density)

        # Spread: move cells away from high density gradient
        flow_x[spread_mask] = -grad_x[spread_mask] / grad_mag[spread_mask] * excess[spread_mask]
        flow_y[spread_mask] = -grad_y[spread_mask] / grad_mag[spread_mask] * excess[spread_mask]

        # Attract: gently pull cells to fill voids
        flow_x[attract_mask] = grad_x[attract_mask] / grad_mag[attract_mask] * np.abs(excess[attract_mask]) * 0.3
        flow_y[attract_mask] = grad_y[attract_mask] / grad_mag[attract_mask] * np.abs(excess[attract_mask]) * 0.3

        return flow_x, flow_y

    def _move_cells_by_flow(self, flow_x: np.ndarray, flow_y: np.ndarray, scale: float):
        """
        Move cells according to the CA-generated flow field.
        Each cell looks up its bin's flow direction and moves accordingly.
        """
        nbx = self.data.num_bins_x
        nby = self.data.num_bins_y
        bw = self.data.bin_width
        bh = self.data.bin_height

        idx = self.data.get_movable_indices()
        cx = self.data.cell_x[idx] + self.data.cell_w[idx] / 2.0
        cy = self.data.cell_y[idx] + self.data.cell_h[idx] / 2.0

        # Find bin indices for each cell
        bi_x = np.clip(((cx - self.data.die_xl) / bw).astype(int), 0, nbx - 1)
        bi_y = np.clip(((cy - self.data.die_yl) / bh).astype(int), 0, nby - 1)

        # Look up flow
        dx = flow_x[bi_y, bi_x] * bw * scale
        dy = flow_y[bi_y, bi_x] * bh * scale

        # Move cells
        self.data.cell_x[idx] += dx
        self.data.cell_y[idx] += dy
        self.data.clip_to_die()

    def _wirelength_refine(self, step_size: float):
        """
        Simple wirelength-driven refinement.
        Move each cell slightly toward the HPWL-weighted center of its connected nets.
        """
        idx = self.data.get_movable_indices()

        # Compute net centers for each movable cell
        target_x = np.zeros(len(idx))
        target_y = np.zeros(len(idx))
        net_count = np.zeros(len(idx))

        movable_set = set(idx.tolist())
        idx_map = {int(v): i for i, v in enumerate(idx)}

        for net in self.data.nets:
            if len(net.pin_indices) < 2:
                continue

            # Compute net center
            px_list = []
            py_list = []
            for pidx in net.pin_indices:
                pin = self.data.pins[pidx]
                ci = pin.cell_index
                px = self.data.cell_x[ci] + self.data.cell_w[ci] / 2.0 + pin.x_offset
                py = self.data.cell_y[ci] + self.data.cell_h[ci] / 2.0 + pin.y_offset
                px_list.append(px)
                py_list.append(py)

            net_cx = np.mean(px_list)
            net_cy = np.mean(py_list)

            # Pull movable cells toward net center
            for pidx in net.pin_indices:
                pin = self.data.pins[pidx]
                ci = pin.cell_index
                if ci in idx_map:
                    mi = idx_map[ci]
                    target_x[mi] += net_cx
                    target_y[mi] += net_cy
                    net_count[mi] += 1

        # Apply movement
        has_nets = net_count > 0
        if np.any(has_nets):
            target_x[has_nets] /= net_count[has_nets]
            target_y[has_nets] /= net_count[has_nets]

            current_x = self.data.cell_x[idx[has_nets]] + self.data.cell_w[idx[has_nets]] / 2.0
            current_y = self.data.cell_y[idx[has_nets]] + self.data.cell_h[idx[has_nets]] / 2.0

            dx = (target_x[has_nets] - current_x) * step_size
            dy = (target_y[has_nets] - current_y) * step_size

            self.data.cell_x[idx[has_nets]] += dx
            self.data.cell_y[idx[has_nets]] += dy
            self.data.clip_to_die()

    def run(self) -> Dict:
        """Execute CA235 placement algorithm."""
        start_time = time.time()
        print("\n" + "=" * 60)
        print("  Cellular Automata CA235 Placement Optimizer")
        print("=" * 60)

        # Start from random or current placement
        print("[CA235] Starting iterative cell spreading...")
        best_hpwl = np.inf
        best_positions = self.data.get_movable_positions().copy()

        for iteration in trange(self.max_iterations, desc="CA235 Spreading", unit="iter"):
            # Compute current density map
            density = compute_density_map_fast(self.data)

            # Apply CA rules multiple times per outer iteration
            for _ in range(self.ca_iterations_per_step):
                flow_x, flow_y = self._ca_rule_235_fast(density)

                # Adaptive scaling: decrease movement over time
                scale = self.move_scale * (1.0 - 0.5 * iteration / self.max_iterations)
                self._move_cells_by_flow(flow_x, flow_y, scale)

                # Recompute density after movement
                density = compute_density_map_fast(self.data)

            # Wirelength refinement
            wl_step = self.wl_refine_weight * (1.0 - 0.3 * iteration / self.max_iterations)
            self._wirelength_refine(wl_step)

            # Evaluate
            hpwl = compute_hpwl(self.data)
            overflow = float(np.sum(np.maximum(0, density - self.target_density)))

            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_positions = self.data.get_movable_positions().copy()

            record = {
                'iteration': iteration,
                'hpwl': hpwl,
                'density_overflow': overflow,
            }
            self.history.append(record)

            if self.callback:
                self.callback(iteration, record)

            if (iteration + 1) % 15 == 0:
                print(f"  Iter {iteration+1:3d}/{self.max_iterations} | "
                      f"HPWL: {hpwl:.0f} | "
                      f"Overflow: {overflow:.4f} | "
                      f"Scale: {scale:.3f}")

        # Restore best
        self.data.set_movable_positions(best_positions)
        self.data.clip_to_die()

        final_density = compute_density_map_fast(self.data)
        final_hpwl = compute_hpwl(self.data)
        final_overflow = float(np.sum(np.maximum(0, final_density - self.target_density)))

        runtime = time.time() - start_time
        print(f"\n[CA235] Done in {runtime:.1f}s")
        print(f"  Final HPWL:     {final_hpwl:.0f}")
        print(f"  Final Overflow: {final_overflow:.4f}")
        print("=" * 60)

        return {
            'hpwl': final_hpwl,
            'density_overflow': final_overflow,
            'density_map': final_density,
            'runtime': runtime,
            'history': self.history,
            'algorithm': 'CA235',
        }
