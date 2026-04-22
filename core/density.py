"""
Density map computation for VLSI placement.
Computes bin-level cell density using overlap-area calculation.
This replaces the PyTorch-based density operator in DREAMPlace.
"""

import numpy as np
from .placement import PlacementData


def compute_density_map(data: PlacementData) -> np.ndarray:
    """
    Compute the cell density map over a grid of bins.

    For each bin, we accumulate the overlap area between each cell
    and the bin. The result is a 2D array of shape (num_bins_y, num_bins_x)
    with values representing the fraction of each bin occupied by cells.

    Returns:
        density: np.ndarray of shape (num_bins_y, num_bins_x)
    """
    nbx = data.num_bins_x
    nby = data.num_bins_y
    bw = data.bin_width
    bh = data.bin_height

    density = np.zeros((nby, nbx), dtype=np.float64)

    cx = data.cell_x
    cy = data.cell_y
    cw = data.cell_w
    ch = data.cell_h

    # Bin area for normalization
    bin_area = bw * bh
    if bin_area == 0:
        return density

    # For each cell, compute which bins it overlaps and accumulate
    # Cell bounding boxes
    cell_xl = cx
    cell_yl = cy
    cell_xh = cx + cw
    cell_yh = cy + ch

    # Bin index ranges for each cell
    bin_xl_idx = np.clip(np.floor((cell_xl - data.die_xl) / bw).astype(int), 0, nbx - 1)
    bin_xh_idx = np.clip(np.floor((cell_xh - data.die_xl) / bw).astype(int), 0, nbx - 1)
    bin_yl_idx = np.clip(np.floor((cell_yl - data.die_yl) / bh).astype(int), 0, nby - 1)
    bin_yh_idx = np.clip(np.floor((cell_yh - data.die_yl) / bh).astype(int), 0, nby - 1)

    for i in range(len(cx)):
        for bxi in range(bin_xl_idx[i], bin_xh_idx[i] + 1):
            for byi in range(bin_yl_idx[i], bin_yh_idx[i] + 1):
                # Bin boundaries
                bx_lo = data.die_xl + bxi * bw
                bx_hi = bx_lo + bw
                by_lo = data.die_yl + byi * bh
                by_hi = by_lo + bh

                # Overlap area
                ox = max(0, min(cell_xh[i], bx_hi) - max(cell_xl[i], bx_lo))
                oy = max(0, min(cell_yh[i], by_hi) - max(cell_yl[i], by_lo))
                overlap = ox * oy

                density[byi, bxi] += overlap / bin_area

    return density


def compute_density_map_fast(data: PlacementData) -> np.ndarray:
    """
    Fast approximate density computation using center-based binning.
    Each cell is assigned to the bin containing its center,
    and its full area is added to that bin.
    Much faster than exact overlap for large cell counts.

    Returns:
        density: np.ndarray of shape (num_bins_y, num_bins_x)
    """
    nbx = data.num_bins_x
    nby = data.num_bins_y
    bw = data.bin_width
    bh = data.bin_height
    bin_area = bw * bh

    density = np.zeros((nby, nbx), dtype=np.float64)

    if bin_area == 0:
        return density

    # Cell centers
    center_x = data.cell_x + data.cell_w / 2.0
    center_y = data.cell_y + data.cell_h / 2.0

    # Bin indices for cell centers
    bi_x = np.clip(((center_x - data.die_xl) / bw).astype(int), 0, nbx - 1)
    bi_y = np.clip(((center_y - data.die_yl) / bh).astype(int), 0, nby - 1)

    # Cell areas
    cell_area = data.cell_w * data.cell_h

    # Accumulate
    np.add.at(density, (bi_y, bi_x), cell_area / bin_area)

    return density


def compute_target_density(data: PlacementData, target_util: float = 1.0) -> np.ndarray:
    """
    Compute the target (uniform) density map.
    This is the ideal density each bin should have.

    Returns:
        target: np.ndarray of shape (num_bins_y, num_bins_x)
    """
    total_movable_area = 0.0
    idx = data.get_movable_indices()
    total_movable_area = np.sum(data.cell_w[idx] * data.cell_h[idx])

    total_bin_area = data.num_bins_x * data.num_bins_y * data.bin_width * data.bin_height
    if total_bin_area == 0:
        target_val = 0.0
    else:
        target_val = total_movable_area / total_bin_area * target_util

    return np.full((data.num_bins_y, data.num_bins_x), target_val, dtype=np.float64)


def density_overflow(density: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the total density overflow.
    Overflow = sum of max(0, density - target) over all bins.
    """
    return float(np.sum(np.maximum(0, density - target)))


def density_penalty(density: np.ndarray, target: np.ndarray) -> float:
    """
    Quadratic density penalty: sum of (density - target)^2 for overflow bins.
    """
    overflow = np.maximum(0, density - target)
    return float(np.sum(overflow ** 2))
