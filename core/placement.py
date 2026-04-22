"""
Placement data structures for DREAMPlace MetaOpt.
Holds all cell, net, pin, and placement region information.
No PyTorch dependency — uses pure NumPy arrays.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class Cell:
    """A single standard cell or macro."""
    name: str
    width: float
    height: float
    is_fixed: bool = False
    # Current placement coordinates (lower-left corner)
    x: float = 0.0
    y: float = 0.0


@dataclass
class Net:
    """A net connecting multiple pins across cells."""
    name: str
    pin_indices: List[int] = field(default_factory=list)     # indices into PlacementData.pins
    cell_indices: List[int] = field(default_factory=list)     # indices into PlacementData.cells


@dataclass
class Pin:
    """A pin on a cell with offset from cell origin."""
    cell_index: int
    x_offset: float = 0.0
    y_offset: float = 0.0


@dataclass
class Row:
    """A placement row in the chip area."""
    y: float
    height: float
    x_start: float
    x_end: float
    site_width: float = 1.0
    num_sites: int = 0


class PlacementData:
    """
    Central data container for the entire placement problem.
    All coordinates are stored as NumPy arrays for fast vectorized math.
    """

    def __init__(self):
        self.name: str = ""
        # Cell lists
        self.cells: List[Cell] = []
        self.nets: List[Net] = []
        self.pins: List[Pin] = []
        self.rows: List[Row] = []

        # Placement region bounds
        self.die_xl: float = 0.0
        self.die_yl: float = 0.0
        self.die_xh: float = 0.0
        self.die_yh: float = 0.0

        # NumPy arrays (populated after loading)
        self.cell_x: Optional[np.ndarray] = None     # shape (N,)
        self.cell_y: Optional[np.ndarray] = None     # shape (N,)
        self.cell_w: Optional[np.ndarray] = None     # width of each cell
        self.cell_h: Optional[np.ndarray] = None     # height of each cell
        self.fixed_mask: Optional[np.ndarray] = None  # bool array

        # Cell name → index mapping
        self.cell_name_to_idx: Dict[str, int] = {}

        # Grid parameters
        self.num_bins_x: int = 64
        self.num_bins_y: int = 64

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    @property
    def num_movable(self) -> int:
        if self.fixed_mask is not None:
            return int(np.sum(~self.fixed_mask))
        return sum(1 for c in self.cells if not c.is_fixed)

    @property
    def num_fixed(self) -> int:
        if self.fixed_mask is not None:
            return int(np.sum(self.fixed_mask))
        return sum(1 for c in self.cells if c.is_fixed)

    @property
    def num_nets(self) -> int:
        return len(self.nets)

    @property
    def die_width(self) -> float:
        return self.die_xh - self.die_xl

    @property
    def die_height(self) -> float:
        return self.die_yh - self.die_yl

    @property
    def bin_width(self) -> float:
        return self.die_width / self.num_bins_x

    @property
    def bin_height(self) -> float:
        return self.die_height / self.num_bins_y

    @property
    def total_cell_area(self) -> float:
        if self.cell_w is not None and self.cell_h is not None:
            movable = ~self.fixed_mask if self.fixed_mask is not None else np.ones(len(self.cells), dtype=bool)
            return float(np.sum(self.cell_w[movable] * self.cell_h[movable]))
        return sum(c.width * c.height for c in self.cells if not c.is_fixed)

    @property
    def utilization(self) -> float:
        die_area = self.die_width * self.die_height
        if die_area == 0:
            return 0.0
        return self.total_cell_area / die_area

    def build_numpy_arrays(self):
        """Convert cell list data into contiguous NumPy arrays for fast computation."""
        n = len(self.cells)
        self.cell_x = np.array([c.x for c in self.cells], dtype=np.float64)
        self.cell_y = np.array([c.y for c in self.cells], dtype=np.float64)
        self.cell_w = np.array([c.width for c in self.cells], dtype=np.float64)
        self.cell_h = np.array([c.height for c in self.cells], dtype=np.float64)
        self.fixed_mask = np.array([c.is_fixed for c in self.cells], dtype=bool)

    def sync_from_numpy(self):
        """Write NumPy positions back to Cell objects."""
        if self.cell_x is not None:
            for i, c in enumerate(self.cells):
                c.x = float(self.cell_x[i])
                c.y = float(self.cell_y[i])

    def get_movable_indices(self) -> np.ndarray:
        """Return indices of movable cells."""
        if self.fixed_mask is not None:
            return np.where(~self.fixed_mask)[0]
        return np.array([i for i, c in enumerate(self.cells) if not c.is_fixed])

    def get_movable_positions(self) -> np.ndarray:
        """Return (N_movable, 2) array of movable cell positions."""
        idx = self.get_movable_indices()
        return np.column_stack([self.cell_x[idx], self.cell_y[idx]])

    def set_movable_positions(self, positions: np.ndarray):
        """Set positions of movable cells from (N_movable, 2) array."""
        idx = self.get_movable_indices()
        self.cell_x[idx] = positions[:, 0]
        self.cell_y[idx] = positions[:, 1]

    def clip_to_die(self):
        """Clip all movable cell positions to stay within die boundaries."""
        idx = self.get_movable_indices()
        self.cell_x[idx] = np.clip(self.cell_x[idx], self.die_xl, self.die_xh - self.cell_w[idx])
        self.cell_y[idx] = np.clip(self.cell_y[idx], self.die_yl, self.die_yh - self.cell_h[idx])

    def random_placement(self, seed: int = 42):
        """Randomly place all movable cells within the die area."""
        rng = np.random.RandomState(seed)
        idx = self.get_movable_indices()
        self.cell_x[idx] = rng.uniform(self.die_xl, self.die_xh - self.cell_w[idx].max(), size=len(idx))
        self.cell_y[idx] = rng.uniform(self.die_yl, self.die_yh - self.cell_h[idx].max(), size=len(idx))

    def center_placement(self):
        """Place all movable cells at the center of the die."""
        idx = self.get_movable_indices()
        cx = (self.die_xl + self.die_xh) / 2.0
        cy = (self.die_yl + self.die_yh) / 2.0
        self.cell_x[idx] = cx - self.cell_w[idx] / 2.0
        self.cell_y[idx] = cy - self.cell_h[idx] / 2.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Benchmark: {self.name}",
            f"Die Area:  ({self.die_xl:.0f}, {self.die_yl:.0f}) → ({self.die_xh:.0f}, {self.die_yh:.0f})",
            f"Die Size:  {self.die_width:.0f} × {self.die_height:.0f}",
            f"Cells:     {self.num_cells:,} total  ({self.num_movable:,} movable, {self.num_fixed:,} fixed)",
            f"Nets:      {self.num_nets:,}",
            f"Utilization: {self.utilization:.1%}",
            f"Grid:      {self.num_bins_x} × {self.num_bins_y} bins",
        ]
        return "\n".join(lines)
