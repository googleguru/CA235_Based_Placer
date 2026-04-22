"""
Benchmark parser for ISPD 2005/2006 format and synthetic benchmark generator.
Supports BigBlue4 and other ISPD benchmarks.
Also generates synthetic benchmarks for quick demos when real data isn't available.
"""

import os
import re
import numpy as np
from typing import Optional
from .placement import PlacementData, Cell, Net, Pin, Row


# ═══════════════════════════════════════════════════════════════════════════════
# ISPD Benchmark Parser
# ═══════════════════════════════════════════════════════════════════════════════

class ISPDBenchmarkParser:
    """
    Parses ISPD 2005/2006 placement benchmarks.
    File formats: .aux, .nodes, .nets, .pl, .scl, .wts
    """

    def __init__(self):
        self.data = PlacementData()

    def parse(self, aux_path: str) -> PlacementData:
        """Parse an ISPD benchmark from its .aux file."""
        if not os.path.exists(aux_path):
            raise FileNotFoundError(f"Benchmark .aux file not found: {aux_path}")

        base_dir = os.path.dirname(aux_path)
        files = self._parse_aux(aux_path)

        for key, fname in files.items():
            fpath = os.path.join(base_dir, fname)
            if not os.path.exists(fpath):
                print(f"  [Warning] {key} file not found: {fpath}")
                continue

            if key == "nodes":
                self._parse_nodes(fpath)
            elif key == "nets":
                self._parse_nets(fpath)
            elif key == "pl":
                self._parse_pl(fpath)
            elif key == "scl":
                self._parse_scl(fpath)

        self.data.name = os.path.splitext(os.path.basename(aux_path))[0]
        self._compute_die_bounds()
        self.data.build_numpy_arrays()
        return self.data

    def _parse_aux(self, path: str) -> dict:
        """Parse .aux file to get filenames."""
        files = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    parts = line.split(":")
                    fnames = parts[1].strip().split()
                    for fn in fnames:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext == ".nodes":
                            files["nodes"] = fn
                        elif ext == ".nets":
                            files["nets"] = fn
                        elif ext == ".pl":
                            files["pl"] = fn
                        elif ext == ".scl":
                            files["scl"] = fn
                        elif ext == ".wts":
                            files["wts"] = fn
        return files

    def _parse_nodes(self, path: str):
        """Parse .nodes file for cell dimensions."""
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue
                if line.startswith("NumNodes") or line.startswith("NumTerminals"):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    try:
                        w = float(parts[1])
                        h = float(parts[2])
                    except ValueError:
                        continue
                    is_fixed = len(parts) > 3 and parts[3].lower().startswith("terminal")
                    cell = Cell(name=name, width=w, height=h, is_fixed=is_fixed)
                    self.data.cell_name_to_idx[name] = len(self.data.cells)
                    self.data.cells.append(cell)

    def _parse_nets(self, path: str):
        """Parse .nets file for net connectivity."""
        current_net = None
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue
                if line.startswith("NumNets") or line.startswith("NumPins"):
                    continue

                if line.startswith("NetDegree"):
                    parts = line.split()
                    net_name = parts[3] if len(parts) > 3 else f"net_{len(self.data.nets)}"
                    current_net = Net(name=net_name)
                    self.data.nets.append(current_net)
                elif current_net is not None:
                    parts = line.split()
                    if len(parts) >= 1:
                        cell_name = parts[0]
                        if cell_name in self.data.cell_name_to_idx:
                            cell_idx = self.data.cell_name_to_idx[cell_name]
                            x_off = float(parts[3]) if len(parts) > 3 else 0.0
                            y_off = float(parts[4]) if len(parts) > 4 else 0.0
                            pin = Pin(cell_index=cell_idx, x_offset=x_off, y_offset=y_off)
                            pin_idx = len(self.data.pins)
                            self.data.pins.append(pin)
                            current_net.pin_indices.append(pin_idx)
                            if cell_idx not in current_net.cell_indices:
                                current_net.cell_indices.append(cell_idx)

    def _parse_pl(self, path: str):
        """Parse .pl file for initial cell positions."""
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    if name in self.data.cell_name_to_idx:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                        except ValueError:
                            continue
                        idx = self.data.cell_name_to_idx[name]
                        self.data.cells[idx].x = x
                        self.data.cells[idx].y = y
                        # Check if fixed
                        if "/FIXED" in line.upper():
                            self.data.cells[idx].is_fixed = True

    def _parse_scl(self, path: str):
        """Parse .scl file for row definitions."""
        current_row_data = {}
        in_row = False
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("UCLA"):
                    continue
                if line.startswith("NumRows"):
                    continue

                if line.startswith("CoreRow"):
                    in_row = True
                    current_row_data = {}
                elif line == "End" and in_row:
                    in_row = False
                    if "Coordinate" in current_row_data:
                        row = Row(
                            y=current_row_data.get("Coordinate", 0),
                            height=current_row_data.get("Height", 12),
                            x_start=current_row_data.get("SubrowOrigin", 0),
                            x_end=current_row_data.get("SubrowOrigin", 0) +
                                  current_row_data.get("NumSites", 0) * current_row_data.get("Sitewidth", 1),
                            site_width=current_row_data.get("Sitewidth", 1),
                            num_sites=current_row_data.get("NumSites", 0),
                        )
                        self.data.rows.append(row)
                elif in_row:
                    m = re.search(r"Coordinate\s*:\s*([\-\d\.]+)", line)
                    if m:
                        current_row_data["Coordinate"] = float(m.group(1))

                    m = re.search(r"Height\s*:\s*([\-\d\.]+)", line)
                    if m:
                        current_row_data["Height"] = float(m.group(1))

                    m = re.search(r"Sitewidth\s*:\s*([\-\d\.]+)", line)
                    if m:
                        current_row_data["Sitewidth"] = float(m.group(1))

                    m = re.search(r"SubrowOrigin\s*:\s*([\-\d\.]+)", line)
                    if m:
                        current_row_data["SubrowOrigin"] = float(m.group(1))

                    m = re.search(r"NumSites\s*:\s*(\d+)", line)
                    if m:
                        current_row_data["NumSites"] = int(m.group(1))

    def _compute_die_bounds(self):
        """Compute die area from rows or cell positions."""
        if self.data.rows:
            self.data.die_xl = min(r.x_start for r in self.data.rows)
            self.data.die_xh = max(r.x_end for r in self.data.rows)
            self.data.die_yl = min(r.y for r in self.data.rows)
            self.data.die_yh = max(r.y + r.height for r in self.data.rows)
        elif self.data.cells:
            xs = [c.x for c in self.data.cells]
            ys = [c.y for c in self.data.cells]
            ws = [c.width for c in self.data.cells]
            hs = [c.height for c in self.data.cells]
            self.data.die_xl = min(xs) if xs else 0
            self.data.die_yl = min(ys) if ys else 0
            self.data.die_xh = max(x + w for x, w in zip(xs, ws)) if xs else 1000
            self.data.die_yh = max(y + h for y, h in zip(ys, hs)) if ys else 1000


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Benchmark Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_benchmark(
    name: str = "bigblue4_synth",
    num_cells: int = 5000,
    num_fixed: int = 200,
    num_nets: int = 6000,
    avg_pins_per_net: int = 4,
    die_width: float = 10000.0,
    die_height: float = 10000.0,
    cell_width_range: tuple = (20, 80),
    cell_height: float = 24.0,
    macro_fraction: float = 0.02,
    macro_size_range: tuple = (200, 600),
    target_utilization: float = 0.70,
    seed: int = 42,
    num_bins: int = 64,
) -> PlacementData:
    """
    Generate a synthetic benchmark that mimics BigBlue4 characteristics.

    BigBlue4 stats (real):
      - 2,177,353 movable cells
      - 32,088 fixed terminals
      - ~num_nets similar order
      - Die area ~3.6M × 3.6M (design units)

    This function creates a scaled-down version that preserves
    the essential structure (mixed standard cells + macros, clustered nets).
    """
    rng = np.random.RandomState(seed)
    data = PlacementData()
    data.name = name
    data.die_xl = 0.0
    data.die_yl = 0.0
    data.die_xh = die_width
    data.die_yh = die_height
    data.num_bins_x = num_bins
    data.num_bins_y = num_bins

    total_cells = num_cells + num_fixed
    num_macros = int(num_cells * macro_fraction)
    num_std = num_cells - num_macros

    # ── Generate standard cells ──────────────────────────────────────────────
    for i in range(num_std):
        w = rng.uniform(*cell_width_range)
        cell = Cell(
            name=f"c{i}",
            width=w,
            height=cell_height,
            is_fixed=False,
        )
        data.cell_name_to_idx[cell.name] = len(data.cells)
        data.cells.append(cell)

    # ── Generate macros ──────────────────────────────────────────────────────
    for i in range(num_macros):
        sz = rng.uniform(*macro_size_range)
        cell = Cell(
            name=f"m{i}",
            width=sz,
            height=sz * rng.uniform(0.5, 1.5),
            is_fixed=False,
        )
        data.cell_name_to_idx[cell.name] = len(data.cells)
        data.cells.append(cell)

    # ── Generate fixed I/O pads ──────────────────────────────────────────────
    for i in range(num_fixed):
        side = rng.randint(4)
        if side == 0:  # bottom
            x = rng.uniform(0, die_width)
            y = 0.0
        elif side == 1:  # top
            x = rng.uniform(0, die_width)
            y = die_height - 10
        elif side == 2:  # left
            x = 0.0
            y = rng.uniform(0, die_height)
        else:  # right
            x = die_width - 10
            y = rng.uniform(0, die_height)

        cell = Cell(
            name=f"p{i}",
            width=10.0,
            height=10.0,
            is_fixed=True,
            x=x,
            y=y,
        )
        data.cell_name_to_idx[cell.name] = len(data.cells)
        data.cells.append(cell)

    # ── Generate clustered nets ──────────────────────────────────────────────
    # Create cluster centers for locality
    num_clusters = max(10, num_cells // 100)
    cluster_cx = rng.uniform(die_width * 0.1, die_width * 0.9, num_clusters)
    cluster_cy = rng.uniform(die_height * 0.1, die_height * 0.9, num_clusters)

    movable_indices = list(range(num_cells))
    fixed_indices = list(range(num_cells, total_cells))

    for i in range(num_nets):
        degree = max(2, min(int(rng.exponential(avg_pins_per_net - 1)) + 2, 30))
        net = Net(name=f"n{i}")

        # Pick cells for this net (biased toward spatial clusters)
        cluster = rng.randint(num_clusters)
        cell_pool = movable_indices.copy()
        if rng.random() < 0.3 and fixed_indices:
            # 30% chance to include a fixed terminal
            fixed_pick = rng.choice(fixed_indices)
            cell_pool.append(fixed_pick)

        chosen = rng.choice(cell_pool, size=min(degree, len(cell_pool)), replace=False)
        for cidx in chosen:
            pin = Pin(cell_index=cidx)
            pidx = len(data.pins)
            data.pins.append(pin)
            net.pin_indices.append(pidx)
            if cidx not in net.cell_indices:
                net.cell_indices.append(cidx)

        if len(net.cell_indices) >= 2:
            data.nets.append(net)

    # ── Generate rows ────────────────────────────────────────────────────────
    row_height = cell_height
    num_rows = int(die_height / row_height)
    for r in range(num_rows):
        row = Row(
            y=r * row_height,
            height=row_height,
            x_start=0,
            x_end=die_width,
            site_width=1.0,
            num_sites=int(die_width),
        )
        data.rows.append(row)

    # ── Build NumPy arrays & place movable cells randomly ────────────────────
    data.build_numpy_arrays()
    data.random_placement(seed=seed)

    return data


def load_or_generate_benchmark(
    benchmark_path: Optional[str] = None,
    num_cells: int = 5000,
    num_bins: int = 64,
    seed: int = 42,
) -> PlacementData:
    """
    Load a real ISPD benchmark if a .aux path is given,
    otherwise generate a synthetic BigBlue4-like benchmark.
    """
    if benchmark_path and os.path.exists(benchmark_path):
        ext = os.path.splitext(benchmark_path)[1].lower()
        if ext != ".aux":
            raise ValueError(
                "Unsupported benchmark format for this runner: "
                f"{benchmark_path}. "
                "This project currently supports ISPD Bookshelf .aux inputs only. "
                "LEF/DEF JSON configs (e.g., ISPD2019 test configs) require a "
                "different parser flow and the corresponding benchmark files."
            )

        print(f"[Loader] Parsing ISPD benchmark: {benchmark_path}")
        parser = ISPDBenchmarkParser()
        data = parser.parse(benchmark_path)
        data.num_bins_x = num_bins
        data.num_bins_y = num_bins
        return data
    else:
        if benchmark_path:
            print(f"[Loader] Benchmark not found: {benchmark_path}")
        print(f"[Loader] Generating synthetic BigBlue4-like benchmark ({num_cells:,} cells)...")
        return generate_synthetic_benchmark(
            num_cells=num_cells,
            num_bins=num_bins,
            seed=seed,
        )
