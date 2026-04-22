"""
DREAMPlace MetaOpt — Single Run Entry Point
============================================
Run all three metaheuristic placement algorithms and launch the GUI.

Usage:
    python run.py                          → Run all algorithms + GUI
    python run.py --algo sbo               → Run only Hybrid SBO
    python run.py --algo pso               → Run only PSO + SBO
    python run.py --algo ca235             → Run only CA235
    python run.py --benchmark path/to.aux  → Use real ISPD benchmark
    python run.py --cells 10000            → Synthetic with 10K cells
    python run.py --no-gui                 → Run without GUI (console only)

No deep learning. No PyTorch. Pure metaheuristic optimization.
"""

import sys
import os
import subprocess
import time
import argparse
import csv
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Get optimal CPU count (respects container limits)
DEFAULT_N_JOBS = min(8, (os.cpu_count() or 1))

# ═══════════════════════════════════════════════════════════════════════════════
# Auto-install dependencies
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_PACKAGES = {
    "numpy":       "numpy>=1.21.0",
    "scipy":       "scipy>=1.7.0",
    "matplotlib":  "matplotlib>=3.5.0",
    "sklearn":     "scikit-learn>=1.0.0",
}


def ensure_dependencies():
    """Check and auto-install missing Python packages."""
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"[Setup] Installing {len(missing)} missing package(s)...")
        for pkg in missing:
            print(f"  -> {pkg}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True, text=True,
            )
        print("[Setup] Dependencies installed.\n")


ensure_dependencies()


# Now import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.benchmark_parser import load_or_generate_benchmark
from core.placement import PlacementData
from core.objectives import PlacementObjective
from core.density import compute_density_map_fast, compute_target_density
from core.potential import compute_potential
from core.field import compute_field, compute_field_magnitude
from core.wirelength import compute_hpwl
from algorithms.hybrid_sbo import HybridSBO
from algorithms.pso_sbo import PSOWithSBO
from algorithms.cellular_automata import CellularAutomataCA235


# ═══════════════════════════════════════════════════════════════════════════════
# Console-only runner (no GUI)
# ═══════════════════════════════════════════════════════════════════════════════

def run_console(data: PlacementData, algo_choice: str = "all", n_jobs: int = 1,
                use_numba: bool = False, use_gpu: bool = False, max_iter: int = None,
                fast_mode: bool = False):
    """Run algorithms and print results to console.
    
    Args:
        data: PlacementData object
        algo_choice: "sbo", "pso", "ca235", or "all"
        n_jobs: Number of parallel workers for objective evaluations
        use_numba: Enable Numba JIT compilation (Phase 2a)
        use_gpu: Enable GPU acceleration via CuPy (Phase 2b)
        max_iter: Override max iterations for algorithms (None = defaults)
        fast_mode: Use shorter defaults for quicker turnaround
    """
    print("\n" + "+" + "=" * 58 + "+")
    print("|   DREAMPlace MetaOpt - Metaheuristic VLSI Placement      |")
    print("|   No Deep Learning - Pure Optimization                   |")
    print("+" + "=" * 58 + "+")
    print(f"|   Execution Mode: n_jobs={n_jobs}, numba={use_numba}, gpu={use_gpu}")
    print("+" + "=" * 58 + "+")
    print()
    print(data.summary())
    print()

    results = {}
    algos_to_run = []

    if algo_choice in ("sbo", "all"):
        algos_to_run.append("sbo")
    if algo_choice in ("pso", "all"):
        algos_to_run.append("pso")
    if algo_choice in ("ca235", "all"):
        algos_to_run.append("ca235")

    num_clusters = min(50, max(5, data.num_movable // 10))

    for algo_name in algos_to_run:
        # Reset to random placement for each algorithm
        data.random_placement(seed=42)
        objective = PlacementObjective(data, use_numba=use_numba, use_gpu=use_gpu)

        # Use provided max_iter or defaults
        if max_iter is not None:
            iter_sbo = max_iter
            iter_pso = max_iter
            iter_ca = max_iter
            pso_particles = 20
        elif fast_mode:
            iter_sbo = 20
            iter_pso = 100
            iter_ca = 30
            pso_particles = 12
        else:
            iter_sbo = 60
            iter_pso = 800  # Changed from 80 to 800
            iter_ca = 100
            pso_particles = 20

        if algo_name == "sbo":
            algo = HybridSBO(data, objective, max_iterations=iter_sbo,
                           num_clusters=num_clusters, n_jobs=n_jobs,
                           use_numba=use_numba, use_gpu=use_gpu)
        elif algo_name == "pso":
            algo = PSOWithSBO(data, objective, max_iterations=iter_pso,
                            num_clusters=num_clusters, num_particles=pso_particles,
                            n_jobs=n_jobs, use_numba=use_numba, use_gpu=use_gpu)
        elif algo_name == "ca235":
            algo = CellularAutomataCA235(data, objective, max_iterations=iter_ca,
                                       n_jobs=n_jobs, use_numba=use_numba, use_gpu=use_gpu)

        result = algo.run()
        # Snapshot final placement state for this algorithm so post-processing
        # visuals remain algorithm-specific when multiple algos are run.
        result['state'] = {
            'cell_x': data.cell_x.copy(),
            'cell_y': data.cell_y.copy(),
            'cell_w': data.cell_w.copy(),
            'cell_h': data.cell_h.copy(),
            'fixed_mask': data.fixed_mask.copy(),
            'die_xl': float(data.die_xl),
            'die_yl': float(data.die_yl),
            'die_width': float(data.die_width),
            'die_height': float(data.die_height),
            'movable_indices': data.get_movable_indices().copy(),
        }
        result['iterations'] = len(result.get('history', []))
        results[algo_name] = result

    # Print comparison
    if len(results) > 1:
        print("\n" + "+" + "=" * 58 + "+")
        print("|               Algorithm Comparison                       |")
        print("+" + "=" * 58 + "+")
        print(f"  {'Algorithm':<22s} {'HPWL':>12s} {'Overflow':>12s} {'Time':>8s}")
        print("  " + "-" * 56)
        for name, res in results.items():
            print(f"  {res['algorithm']:<22s} "
                  f"{res['hpwl']:>12,.0f} "
                  f"{res['density_overflow']:>12.6f} "
                  f"{res['runtime']:>7.1f}s")

    # Save density maps
    save_results(data, results)

    if len(results) > 1:
        save_detailed_comparison(data, results)

    return results


def save_results(data: PlacementData, results: dict, output_dir: str = None, output_prefix: str = None):
    """Save result plots to the results/ directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if output_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    else:
        results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)

    def make_bigblue_overlay(data_obj: PlacementData, density_map, ex_field, ey_field):
        """Build a BigBlue-like panel (red blocks + blue electric traces)."""
        h, w = density_map.shape
        occupancy = np.zeros((h, w), dtype=np.float64)

        mov_idx = data_obj.get_movable_indices()
        if len(mov_idx) > 0:
            cx = data_obj.cell_x[mov_idx] + data_obj.cell_w[mov_idx] / 2.0
            cy = data_obj.cell_y[mov_idx] + data_obj.cell_h[mov_idx] / 2.0
            ix = np.clip(((cx - data_obj.die_xl) / max(data_obj.die_width, 1e-9) * w).astype(int), 0, w - 1)
            iy = np.clip(((cy - data_obj.die_yl) / max(data_obj.die_height, 1e-9) * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 1.0)

        fixed_idx = np.where(data_obj.fixed_mask)[0]
        if len(fixed_idx) > 0:
            fx = data_obj.cell_x[fixed_idx] + data_obj.cell_w[fixed_idx] / 2.0
            fy = data_obj.cell_y[fixed_idx] + data_obj.cell_h[fixed_idx] / 2.0
            ix = np.clip(((fx - data_obj.die_xl) / max(data_obj.die_width, 1e-9) * w).astype(int), 0, w - 1)
            iy = np.clip(((fy - data_obj.die_yl) / max(data_obj.die_height, 1e-9) * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 3.0)

        occ_max = float(np.max(occupancy))
        if occ_max > 0:
            occupancy = occupancy / occ_max

        field_mag = np.sqrt(ex_field ** 2 + ey_field ** 2)
        fmax = float(np.max(field_mag))
        if fmax > 0:
            field_mag = field_mag / fmax

        panel = np.ones((h, w, 3), dtype=np.float64)
        panel[..., 0] = 1.0
        panel[..., 1] = 1.0 - 0.55 * occupancy
        panel[..., 2] = 1.0 - 0.55 * occupancy
        panel[..., 0] = np.clip(panel[..., 0] - 0.45 * field_mag, 0.0, 1.0)
        panel[..., 1] = np.clip(panel[..., 1] - 0.65 * field_mag, 0.0, 1.0)
        return panel

    def make_bigblue_overlay_from_state(state: dict, density_map, ex_field, ey_field):
        """BigBlue-like overlay rendered from a saved algorithm snapshot."""
        h, w = density_map.shape
        occupancy = np.zeros((h, w), dtype=np.float64)

        cell_x = state['cell_x']
        cell_y = state['cell_y']
        cell_w = state['cell_w']
        cell_h = state['cell_h']
        fixed_mask = state['fixed_mask']
        movable_idx = state['movable_indices']

        die_xl = state['die_xl']
        die_yl = state['die_yl']
        die_w = max(state['die_width'], 1e-9)
        die_h = max(state['die_height'], 1e-9)

        if len(movable_idx) > 0:
            cx = cell_x[movable_idx] + cell_w[movable_idx] / 2.0
            cy = cell_y[movable_idx] + cell_h[movable_idx] / 2.0
            ix = np.clip(((cx - die_xl) / die_w * w).astype(int), 0, w - 1)
            iy = np.clip(((cy - die_yl) / die_h * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 1.0)

        fixed_idx = np.where(fixed_mask)[0]
        if len(fixed_idx) > 0:
            fx = cell_x[fixed_idx] + cell_w[fixed_idx] / 2.0
            fy = cell_y[fixed_idx] + cell_h[fixed_idx] / 2.0
            ix = np.clip(((fx - die_xl) / die_w * w).astype(int), 0, w - 1)
            iy = np.clip(((fy - die_yl) / die_h * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 3.0)

        occ_max = float(np.max(occupancy))
        if occ_max > 0:
            occupancy = occupancy / occ_max

        field_mag = np.sqrt(ex_field ** 2 + ey_field ** 2)
        fmax = float(np.max(field_mag))
        if fmax > 0:
            field_mag = field_mag / fmax

        panel = np.ones((h, w, 3), dtype=np.float64)
        panel[..., 0] = 1.0
        panel[..., 1] = 1.0 - 0.55 * occupancy
        panel[..., 2] = 1.0 - 0.55 * occupancy
        panel[..., 0] = np.clip(panel[..., 0] - 0.45 * field_mag, 0.0, 1.0)
        panel[..., 1] = np.clip(panel[..., 1] - 0.65 * field_mag, 0.0, 1.0)
        return panel

    for name, result in results.items():
        density = result.get('density_map', compute_density_map_fast(data))
        potential = compute_potential(data, density, method='dct')
        Ex, Ey = compute_field(potential, data)
        field_mag = compute_field_magnitude(Ex, Ey)
        state = result.get('state')

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), facecolor='#020817')
        fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.08, wspace=0.12)

        if state is not None:
            overlay = make_bigblue_overlay_from_state(state, density, Ex, Ey)
        else:
            overlay = make_bigblue_overlay(data, density, Ex, Ey)
        axes[0].imshow(overlay, origin='lower', interpolation='nearest')
        axes[0].set_title("Bigblue4", color='white', fontsize=13, fontweight='bold')
        iter_count = result.get('iterations', len(result.get('history', [])))
        if iter_count:
            axes[0].text(0.97, 0.03, f"Iter: {iter_count}", transform=axes[0].transAxes,
                         ha='right', va='bottom', color='#2f6df6', fontsize=11, fontweight='bold')

        axes[1].imshow(density, cmap='Greys', origin='lower', interpolation='nearest')
        axes[1].set_title("Density Map", color='white', fontsize=13, fontweight='bold')

        axes[2].imshow(potential, cmap='gray', origin='lower', interpolation='nearest')
        axes[2].set_title("Electric Potential", color='white', fontsize=13, fontweight='bold')

        axes[3].imshow(field_mag, cmap='gray', origin='lower', interpolation='nearest')
        axes[3].set_title("Electric Field", color='white', fontsize=13, fontweight='bold')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#0a1020')
            for spine in ax.spines.values():
                spine.set_color('#2b3448')
                spine.set_linewidth(1.0)

        if output_prefix:
            filename = f"{output_prefix}_{name}_result.png"
        else:
            filename = f"{name}_result.png"
        outpath = os.path.join(results_dir, filename)
        fig.savefig(outpath, dpi=220, facecolor='#0d1117', bbox_inches='tight')
        plt.close(fig)
        print(f"  [Saved] {outpath}")


def save_detailed_comparison(data: PlacementData, results: dict, output_dir: str = None, output_prefix: str = None):
    """Save a detailed multi-algorithm comparison image + CSV metrics table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if output_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    else:
        results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)

    def make_overlay(state: dict, density_map, ex_field, ey_field):
        h, w = density_map.shape
        occupancy = np.zeros((h, w), dtype=np.float64)

        cell_x = state['cell_x']
        cell_y = state['cell_y']
        cell_w = state['cell_w']
        cell_h = state['cell_h']
        fixed_mask = state['fixed_mask']
        movable_idx = state['movable_indices']

        die_xl = state['die_xl']
        die_yl = state['die_yl']
        die_w = max(state['die_width'], 1e-9)
        die_h = max(state['die_height'], 1e-9)

        if len(movable_idx) > 0:
            cx = cell_x[movable_idx] + cell_w[movable_idx] / 2.0
            cy = cell_y[movable_idx] + cell_h[movable_idx] / 2.0
            ix = np.clip(((cx - die_xl) / die_w * w).astype(int), 0, w - 1)
            iy = np.clip(((cy - die_yl) / die_h * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 1.0)

        fixed_idx = np.where(fixed_mask)[0]
        if len(fixed_idx) > 0:
            fx = cell_x[fixed_idx] + cell_w[fixed_idx] / 2.0
            fy = cell_y[fixed_idx] + cell_h[fixed_idx] / 2.0
            ix = np.clip(((fx - die_xl) / die_w * w).astype(int), 0, w - 1)
            iy = np.clip(((fy - die_yl) / die_h * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 3.0)

        occ_max = float(np.max(occupancy))
        if occ_max > 0:
            occupancy = occupancy / occ_max

        field_mag = np.sqrt(ex_field ** 2 + ey_field ** 2)
        fmax = float(np.max(field_mag))
        if fmax > 0:
            field_mag = field_mag / fmax

        panel = np.ones((h, w, 3), dtype=np.float64)
        panel[..., 0] = 1.0
        panel[..., 1] = 1.0 - 0.55 * occupancy
        panel[..., 2] = 1.0 - 0.55 * occupancy
        panel[..., 0] = np.clip(panel[..., 0] - 0.45 * field_mag, 0.0, 1.0)
        panel[..., 1] = np.clip(panel[..., 1] - 0.65 * field_mag, 0.0, 1.0)
        return panel

    ordered_keys = [k for k in ("sbo", "pso", "ca235") if k in results]
    if not ordered_keys:
        ordered_keys = list(results.keys())

    rows = len(ordered_keys)
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4.2 * rows), facecolor='#020817')
    if rows == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.03, right=0.985, top=0.93, bottom=0.07, wspace=0.10, hspace=0.28)
    fig.suptitle("Detailed Algorithm Comparison", color='white', fontsize=17, fontweight='bold')

    # CSV metrics table
    if output_prefix:
        csv_name = f"{output_prefix}_comparison_metrics.csv"
    else:
        csv_name = "algorithm_comparison_metrics.csv"
    csv_path = os.path.join(results_dir, csv_name)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm_key", "algorithm", "hpwl", "density_overflow", "runtime_s", "iterations"])
        for key in ordered_keys:
            r = results[key]
            writer.writerow([
                key,
                r.get('algorithm', key),
                f"{r.get('hpwl', 0):.6f}",
                f"{r.get('density_overflow', 0):.6f}",
                f"{r.get('runtime', 0):.3f}",
                int(r.get('iterations', len(r.get('history', []))))
            ])

    for row_idx, key in enumerate(ordered_keys):
        result = results[key]
        state = result.get('state')

        density = result.get('density_map', compute_density_map_fast(data))
        potential = compute_potential(data, density, method='dct')
        Ex, Ey = compute_field(potential, data)
        field_mag = compute_field_magnitude(Ex, Ey)

        if state is not None:
            overlay = make_overlay(state, density, Ex, Ey)
        else:
            overlay = np.stack([density, density, density], axis=-1)

        axes[row_idx, 0].imshow(overlay, origin='lower', interpolation='nearest')
        axes[row_idx, 1].imshow(density, cmap='Greys', origin='lower', interpolation='nearest')
        axes[row_idx, 2].imshow(potential, cmap='gray', origin='lower', interpolation='nearest')
        axes[row_idx, 3].imshow(field_mag, cmap='gray', origin='lower', interpolation='nearest')

        iter_count = int(result.get('iterations', len(result.get('history', []))))
        title_left = (
            f"{result.get('algorithm', key)} | "
            f"HPWL: {result.get('hpwl', 0):,.0f} | "
            f"Overflow: {result.get('density_overflow', 0):.4f} | "
            f"Time: {result.get('runtime', 0):.1f}s | "
            f"Iter: {iter_count}"
        )
        axes[row_idx, 0].set_title(title_left, color='white', fontsize=11, fontweight='bold')
        axes[row_idx, 1].set_title("Density Map", color='white', fontsize=12, fontweight='bold')
        axes[row_idx, 2].set_title("Electric Potential", color='white', fontsize=12, fontweight='bold')
        axes[row_idx, 3].set_title("Electric Field", color='white', fontsize=12, fontweight='bold')

        for col in range(4):
            ax = axes[row_idx, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#0a1020')
            for spine in ax.spines.values():
                spine.set_color('#2b3448')
                spine.set_linewidth(1.0)

    if output_prefix:
        out_name = f"{output_prefix}_detailed_comparison.png"
    else:
        out_name = "detailed_algorithm_comparison.png"
    outpath = os.path.join(results_dir, out_name)
    fig.savefig(outpath, dpi=240, facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)

    print(f"  [Saved] {outpath}")
    print(f"  [Saved] {csv_path}")


def collect_benchmark_aux_files(bench_root: str, include_variants: bool = False):
    """Collect benchmark .aux files from a benchmark root folder."""
    pattern = os.path.join(bench_root, "**", "*.aux")
    aux_files = sorted(glob.glob(pattern, recursive=True))
    if include_variants:
        return aux_files

    filtered = []
    for path in aux_files:
        stem = os.path.basename(path).lower()
        if ".dp.aux" in stem or ".eplace.aux" in stem:
            continue
        filtered.append(path)
    return filtered


def run_all_benchmarks(args):
    """Run selected algorithm(s) across all discovered benchmark .aux files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_root = args.bench_root or os.path.join(script_dir, "benchmarks", "ispd2005")

    if not os.path.isdir(bench_root):
        print(f"[Batch] Benchmark root not found: {bench_root}")
        return 1

    aux_files = collect_benchmark_aux_files(bench_root, include_variants=args.include_aux_variants)
    if not aux_files:
        print(f"[Batch] No .aux files found under: {bench_root}")
        return 1

    if args.max_benchmarks > 0:
        aux_files = aux_files[:args.max_benchmarks]

    output_dir = os.path.join(script_dir, "results", "all_benchmarks")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Batch] Found {len(aux_files)} benchmark(s)")
    print(f"[Batch] Output folder: {output_dir}")

    failed = []
    total_start = time.time()

    for idx, aux_path in enumerate(aux_files, start=1):
        bench_name = os.path.splitext(os.path.basename(aux_path))[0]
        print("\n" + "=" * 78)
        print(f"[Batch] {idx}/{len(aux_files)}  Running benchmark: {bench_name}")
        print(f"[Batch] AUX: {aux_path}")
        print("=" * 78)

        try:
            data = load_or_generate_benchmark(
                benchmark_path=aux_path,
                num_cells=args.cells,
                num_bins=args.bins,
                seed=args.seed,
            )
            results = run_console(data, args.algo)
            save_results(data, results, output_dir=output_dir, output_prefix=bench_name)
        except Exception as ex:
            print(f"[Batch][ERROR] {bench_name}: {ex}")
            failed.append((bench_name, str(ex)))

    elapsed = time.time() - total_start
    print("\n" + "=" * 78)
    print(f"[Batch] Completed in {elapsed:.1f}s")
    print(f"[Batch] Success: {len(aux_files) - len(failed)} / {len(aux_files)}")
    if failed:
        print("[Batch] Failed benchmarks:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
    print("=" * 78)
    return 0 if not failed else 2


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DREAMPlace MetaOpt - Metaheuristic VLSI Placement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           Run all algorithms + open GUI
  python run.py --algo sbo                Run Hybrid SBO only
  python run.py --algo pso                Run PSO+SBO only
  python run.py --algo ca235              Run CA235 only
  python run.py --cells 10000             Use 10K synthetic cells
  python run.py --benchmark bigblue4.aux  Use real ISPD benchmark
  python run.py --batch --n_jobs 8        Batch mode with 8 parallel workers (no GUI)
  python run.py --no-gui                  Console only, no GUI window
        """
    )

    parser.add_argument("--algo", choices=["sbo", "pso", "ca235", "all"],
                       default="all", help="Algorithm to run (default: all)")
    parser.add_argument("--benchmark", type=str, default=None,
                       help="Path to ISPD benchmark .aux file")
    parser.add_argument("--cells", type=int, default=3000,
                       help="Number of synthetic cells (default: 3000)")
    parser.add_argument("--bins", type=int, default=64,
                       help="Number of grid bins per axis (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--batch", action="store_true",
                       help="Run in batch mode (no GUI, skip Tkinter)")
    parser.add_argument("--n_jobs", type=int, default=DEFAULT_N_JOBS,
                       help=f"Number of parallel workers for objective evaluations (default: {DEFAULT_N_JOBS})")
    parser.add_argument("--use_numba", action="store_true",
                       help="Enable Numba JIT compilation for compute kernels (Phase 2a)")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Enable GPU acceleration via CuPy (Phase 2b, requires NVIDIA GPU)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run without GUI (console output only) [deprecated, use --batch]")
    parser.add_argument("--all-benchmarks", action="store_true",
                       help="Run all discovered benchmark .aux files and save outputs")
    parser.add_argument("--bench-root", type=str, default=None,
                       help="Benchmark root folder for --all-benchmarks (default: benchmarks/ispd2005)")
    parser.add_argument("--include-aux-variants", action="store_true",
                       help="Include .dp.aux and .eplace.aux in --all-benchmarks")
    parser.add_argument("--max-benchmarks", type=int, default=0,
                       help="Limit number of benchmarks in --all-benchmarks (0 means all)")
    parser.add_argument("--max-iter", type=int, default=None,
                       help="Max iterations for algorithms (default: SBO=60, PSO=800, CA235=100)")
    parser.add_argument("--fast", action="store_true",
                       help="Use reduced defaults for faster comparison runs")

    args = parser.parse_args()
    
    # Backward compatibility: --no-gui → --batch
    if args.no_gui:
        args.batch = True

    if args.all_benchmarks:
        rc = run_all_benchmarks(args)
        raise SystemExit(rc)

    # Load or generate benchmark
    data = load_or_generate_benchmark(
        benchmark_path=args.benchmark,
        num_cells=args.cells,
        num_bins=args.bins,
        seed=args.seed,
    )

    if args.batch:
        # Batch/Console mode
        print(f"[DREAMPlace MetaOpt] Batch mode - n_jobs={args.n_jobs}")
        run_console(data, args.algo, n_jobs=args.n_jobs, 
                   use_numba=args.use_numba, use_gpu=args.use_gpu,
                   max_iter=args.max_iter, fast_mode=args.fast)
    else:
        # GUI mode
        print("\n[DREAMPlace MetaOpt] Starting GUI...")
        from gui.visualizer import PlacementVisualizer
        import tkinter as tk

        root = tk.Tk()
        app = PlacementVisualizer(root)
        app.set_data(data)
        app.run()


if __name__ == "__main__":
    main()
