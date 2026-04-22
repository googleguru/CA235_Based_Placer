#!/usr/bin/env python
"""
Quick test of Phase 1: batch mode with --n_jobs multiprocessing
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.benchmark_parser import load_or_generate_benchmark
from core.placement import PlacementData

print("[Phase 1 Test] Testing batch mode with n_jobs parameter...")

# Small synthetic benchmark for quick test
print("\n1. Loading synthetic benchmark (1000 cells)...")
data = load_or_generate_benchmark(
    benchmark_path=None,
    num_cells=1000,
    num_bins=32,
    seed=42
)
print(f"   ✓ Loaded: {data.num_cells} cells, {data.num_movable} movable")

# Test PSO with n_jobs > 1
print("\n2. Testing PSO+SBO with n_jobs=2...")
from algorithms.pso_sbo import PSOWithSBO
from core.objectives import PlacementObjective

pso_algo = PSOWithSBO(
    data=data,
    objective=PlacementObjective(data),
    num_clusters=10,
    num_particles=10,
    max_iterations=3,  # Very short test
    n_jobs=2,
    use_numba=False,
    use_gpu=False
)
print(f"   ✓ PSO initialized with n_jobs={pso_algo.n_jobs}")

# Test SBO with n_jobs > 1
print("\n3. Testing Hybrid SBO with n_jobs=2...")
from algorithms.hybrid_sbo import HybridSBO

sbo_algo = HybridSBO(
    data=data,
    objective=PlacementObjective(data),
    num_clusters=10,
    initial_samples=5,
    max_iterations=2,  # Very short test
    n_jobs=2,
    use_numba=False,
    use_gpu=False
)
print(f"   ✓ SBO initialized with n_jobs={sbo_algo.n_jobs}")

# Test CA with n_jobs > 1
print("\n4. Testing Cellular Automata CA235 with n_jobs=2...")
from algorithms.cellular_automata import CellularAutomataCA235

ca_algo = CellularAutomataCA235(
    data=data,
    objective=PlacementObjective(data),
    max_iterations=3,  # Very short test
    n_jobs=2,
    use_numba=False,
    use_gpu=False
)
print(f"   ✓ CA235 initialized with n_jobs={ca_algo.n_jobs}")

print("\n" + "=" * 60)
print("✓ Phase 1 initialization test PASSED")
print("=" * 60)
print("All algorithm classes support n_jobs parameter!")
print("Ready to test full optimization runs with multiprocessing.")
print("\nNext steps:")
print("  1. Build Docker image: docker build -t dreamplace:v1 .")
print("  2. Test batch mode: docker run --cpus=4 dreamplace:v1 \\")
print("       python run.py --batch --n_jobs 4 --benchmark benchmarks/ispd2005/adaptec1/adaptec1.aux")
print("  3. Compare with sequential: python run.py --batch --n_jobs 1 --benchmark adaptec1.aux")
print("=" * 60)
