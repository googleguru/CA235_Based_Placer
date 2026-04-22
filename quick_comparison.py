"""
Quick comparison generator - runs all 3 algorithms with reasonable iterations
and creates a beautiful tabulation + visualization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from pathlib import Path

from core.benchmark_parser import load_or_generate_benchmark
from core.placement import PlacementData
from core.objectives import PlacementObjective
from core.density import compute_density_map_fast
from core.potential import compute_potential
from core.field import compute_field, compute_field_magnitude
from algorithms.hybrid_sbo import HybridSBO
from algorithms.pso_sbo import PSOWithSBO
from algorithms.cellular_automata import CellularAutomataCA235


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


def run_comparison(benchmark_path="benchmarks/ispd2005/adaptec1/adaptec1.aux"):
    """Run all 3 algorithms and generate comparison."""
    
    # Load benchmark
    data = load_or_generate_benchmark(benchmark_path=benchmark_path, seed=42)
    
    print("\n" + "="*80)
    print("DREAMPlace MetaOpt - ALGORITHM COMPARISON")
    print("="*80)
    print(data.summary())
    
    algorithms = [
        ("SBO",    HybridSBO,           {"max_iterations": 60,   "num_clusters": 50}),
        ("PSO",    PSOWithSBO,          {"max_iterations": 800,  "num_clusters": 50, "num_particles": 20}),
        ("CA235",  CellularAutomataCA235, {"max_iterations": 100}),
    ]
    
    all_results = {}
    table_data = []
    
    for algo_name, algo_class, params in algorithms:
        print(f"\n[{algo_name}] Running optimization...")
        
        # Reset placement
        data.random_placement(seed=42)
        objective = PlacementObjective(data)
        
        # Run algorithm
        algo = algo_class(data, objective, n_jobs=1, **params)
        result = algo.run()
        
        all_results[algo_name] = result
        
        # Collect data for table
        iter_count = result.get('iterations', len(result.get('history', [])))
        table_data.append([
            algo_name,
            f"{result['hpwl']:,.0f}",
            f"{result['density_overflow']:.4f}",
            f"{result['runtime']:.1f}",
            iter_count,
        ])
    
    # Print comparison table
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON TABLE")
    print("="*80)
    headers = ["Algorithm", "HPWL", "Density Overflow", "Time (s)", "Iterations"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80 + "\n")
    
    # Create comparison visualization (3 algorithms x 4 panels)
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), facecolor='#020817')
    fig.subplots_adjust(left=0.04, right=0.985, top=0.94, bottom=0.06, wspace=0.15, hspace=0.35)
    
    title_text = "DREAMPlace MetaOpt - Algorithm Comparison"
    if len(benchmark_path) > 0:
        benchmark_name = Path(benchmark_path).stem
        title_text += f" ({benchmark_name})"
    fig.suptitle(title_text, fontsize=16, fontweight='bold', color='white', y=0.98)
    
    for row_idx, (algo_name, result) in enumerate(all_results.items()):
        # Get visualization data
        density = compute_density_map_fast(data)
        potential = compute_potential(data, density, method='dct')
        Ex, Ey = compute_field(potential, data)
        field_mag = compute_field_magnitude(Ex, Ey)
        
        # Get metrics
        iter_count = result.get('iterations', len(result.get('history', [])))
        hpwl = result['hpwl']
        overflow = result['density_overflow']
        runtime = result['runtime']
        
        # Row title with metrics
        row_title = f"{algo_name}  |  HPWL: {hpwl:,.0f}  |  Overflow: {overflow:.4f}  |  Time: {runtime:.0f}s  |  Iter: {iter_count}"
        
        # Column 0: Bigblue4-style
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
        
        # Format axes
        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            axes[row_idx, col_idx].set_facecolor('#0a1020')
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_color('#2b3448')
                spine.set_linewidth(1.0)
    
    # Save comparison
    os.makedirs("results", exist_ok=True)
    outpath = os.path.join("results", "COMPARISON_all_algorithms.png")
    fig.savefig(outpath, dpi=220, facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ [Saved] {outpath}")
    return outpath


if __name__ == "__main__":
    run_comparison()
