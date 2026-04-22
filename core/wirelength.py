"""
Wirelength computation for VLSI placement.
Uses Half-Perimeter Wirelength (HPWL) — the standard metric.

HPWL for a net = (max_x - min_x) + (max_y - min_y) over all pins.
"""

import numpy as np
from .placement import PlacementData


def compute_hpwl(data: PlacementData) -> float:
    """
    Compute total Half-Perimeter Wirelength over all nets.

    Returns:
        Total HPWL (float)
    """
    total_hpwl = 0.0

    for net in data.nets:
        if len(net.pin_indices) < 2:
            continue

        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf

        for pidx in net.pin_indices:
            pin = data.pins[pidx]
            ci = pin.cell_index
            px = data.cell_x[ci] + data.cell_w[ci] / 2.0 + pin.x_offset
            py = data.cell_y[ci] + data.cell_h[ci] / 2.0 + pin.y_offset

            if px < min_x:
                min_x = px
            if px > max_x:
                max_x = px
            if py < min_y:
                min_y = py
            if py > max_y:
                max_y = py

        total_hpwl += (max_x - min_x) + (max_y - min_y)

    return total_hpwl


def compute_hpwl_per_net(data: PlacementData) -> np.ndarray:
    """
    Compute HPWL for each individual net.

    Returns:
        Array of HPWL values, one per net.
    """
    hpwl_list = np.zeros(len(data.nets), dtype=np.float64)

    for ni, net in enumerate(data.nets):
        if len(net.pin_indices) < 2:
            continue

        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf

        for pidx in net.pin_indices:
            pin = data.pins[pidx]
            ci = pin.cell_index
            px = data.cell_x[ci] + data.cell_w[ci] / 2.0 + pin.x_offset
            py = data.cell_y[ci] + data.cell_h[ci] / 2.0 + pin.y_offset

            min_x = min(min_x, px)
            max_x = max(max_x, px)
            min_y = min(min_y, py)
            max_y = max(max_y, py)

        hpwl_list[ni] = (max_x - min_x) + (max_y - min_y)

    return hpwl_list
