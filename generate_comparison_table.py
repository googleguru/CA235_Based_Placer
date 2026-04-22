"""Generate a comparison table and visualization for all algorithms."""

import sys
import os
import json
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.benchmark_parser import load_or_generate_benchmark
from core.placement import PlacementData
from core.objectives import PlacementObjective
from core.density import compute_density_map_fast, compute_target_density
from core.potential import compute_potential
from core.field import compute_field, compute_field_magnitude


def run_all_algos(benchmark_path, max_iter=800):
    """Run all 3 algorithms and collect results."""
    # Load benchmark
    data = load_or_generate_benchmark(
        benchmark_path=benchmark_path,
        seed=42,
    )
    
    print("\n" + "=" * 70)
    print("RUNNING ALL 3 ALGORITHMS FOR COMPARISON")
    print("=" * 70)
    
    results_summary = {}
    all_results = {}
    
    # Import algorithms
    from algorithms.hybrid_sbo import HybridSBO
    from algorithms.pso_sbo import PSOWithSBO
    from algorithms.cellular_automata import CellularAutomataCA235
    
    algorithms = [
        ("SBO", HybridSBO, {"max_iterations": 60}),
        ("PSO", PSOWithSBO, {"max_iterations": max_iter, "num_particles": 20}),
        ("CA235", CellularAutomataCA235, {"max_iterations": 100}),
    ]
    
    for algo_name, algo_class, params in algorithms:
        print(f"\n[{algo_name}] Running...")
        data.random_placement(seed=42)
        objective = PlacementObjective(data)
        
        algo = algo_class(data, objective, **params, n_jobs=1)
        result = algo.run()
        
        all_results[algo_name] = result
        results_summary[algo_name] = {
            "Algorithm": algo_name,
            "HPWL": f"{result['hpwl']:,.0f}",
            "Overflow": f"{result['density_overflow']:.4f}",
            "Runtime (s)": f"{result['runtime']:.1f}",
            "Iterations": result.get('iterations', len(result.get('history', []))),
        }
    
    return data, all_results, results_summary


def print_comparison_table(results_summary):
    """Print formatted comparison table."""
    headers = ["Algorithm", "HPWL", "Overflow", "Runtime (s)", "Iterations"]
    rows = []
    
    for algo_name in ["SBO", "PSO", "CA235"]:
        if algo_name in results_summary:
            r = results_summary[algo_name]
            rows.append([
                r["Algorithm"],
                r["HPWL"],
                r["Overflow"],
                r["Runtime (s)"],
                r["Iterations"],
            ])
    
    print("\n" + "=" * 90)
    print("ALGORITHM COMPARISON TABLE")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 90)


def make_bigblue_overlay(data_obj: PlacementData, density_map, ex_field, ey_field):
    """Build a BigBlue-like panel (red blocks + blue electric traces)."""
    h, w = density_map.shape
    occupancy = np.zeros((h, w), dtype=np.float64)
    
    movable_pos = data_obj.get_movable_positions()
    bin_width = (data_obj.die_ub[0] - data_obj.die_lb[0]) / w
    bin_height = (data_obj.die_ub[1] - data_obj.die_lb[1]) / h
    
    for x, y in movable_pos:
        gx = int((x - data_obj.die_lb[0]) / bin_width)
        gy = int((y - data_obj.die_lb[1]) / bin_height)
        gx, gy = np.clip(gx, 0, w-1), np.clip(gy, 0, h-1)
        occupancy[gy, gx] += 1.0
    
    occupancy = np.clip(occupancy / (occupancy.max() + 1e-6), 0.0, 1.0)
    
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


def create_comparison_visualization(data, all_results, results_dir="results"):
    """Create side-by-side visualization of all 3 algorithms."""
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), facecolor='#020817')
    fig.subplots_adjust(left=0.04, right=0.985, top=0.94, bottom=0.06, wspace=0.15, hspace=0.35)
    
    fig.suptitle("DREAMPlace MetaOpt - Algorithm Comparison (Adaptec1, 211K Cells)", 
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    
    for row_idx, (algo_name, result) in enumerate(all_results.items()):
        # Recompute density/potential for visualization
        density = compute_density_map_fast(data)
        potential = compute_potential(data, density, method='dct')
        Ex, Ey = compute_field(potential, data)
        field_mag = compute_field_magnitude(Ex, Ey)
        
        # Row title
        iter_count = result.get('iterations', len(result.get('history', [])))
        hpwl = result['hpwl']
        overflow = result['density_overflow']
        runtime = result['runtime']
        
        row_title = f"{algo_name} | HPWL: {hpwl:,.0f} | Overflow: {overflow:.4f} | Time: {runtime:.0f}s | Iter: {iter_count}"
        
        # Column 0: Bigblue overlay
        axes[row_idx, 0].imshow(make_bigblue_overlay(data, density, Ex, Ey), 
                               origin='lower', interpolation='nearest')
        axes[row_idx, 0].set_title(row_title, color='white', fontsize=11, fontweight='bold', pad=10)
        
        # Column 1: Density Map
        axes[row_idx, 1].imshow(density, cmap='Greys', origin='lower', interpolation='nearest')
        axes[row_idx, 1].set_title("Density Map", color='white', fontsize=10, fontweight='bold')
        
        # Column 2: Electric Potential
        axes[row_idx, 2].imshow(potential, cmap='gray', origin='lower', interpolation='nearest')
        axes[row_idx, 2].set_title("Electric Potential", color='white', fontsize=10, fontweight='bold')
        
        # Column 3: Electric Field
        axes[row_idx, 3].imshow(field_mag, cmap='gray', origin='lower', interpolation='nearest')
        axes[row_idx, 3].set_title("Electric Field", color='white', fontsize=10, fontweight='bold')
        
        # Format all axes
        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            axes[row_idx, col_idx].set_facecolor('#0a1020')
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_color('#2b3448')
                spine.set_linewidth(1.0)
    
    # Save comparison visualization
    outpath = os.path.join(results_dir, "algorithm_comparison.png")
    fig.savefig(outpath, dpi=220, facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ [Saved] {outpath}")


if __name__ == "__main__":
    benchmark_path = "benchmarks/ispd2005/adaptec1/adaptec1.aux"
    
    # Run all algorithms
    data, all_results, results_summary = run_all_algos(benchmark_path, max_iter=800)
    
    # Print comparison table
    print_comparison_table(results_summary)
    
    # Create comparison visualization
    create_comparison_visualization(data, all_results)
    
    print("\n✓ Comparison complete!")
