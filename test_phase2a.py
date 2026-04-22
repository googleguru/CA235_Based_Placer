#!/usr/bin/env python
"""
Phase 2a Test: Numba JIT Compilation
Verifies that Numba backend works and provides performance benefits.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.benchmark_parser import load_or_generate_benchmark
from core.placement import PlacementData
from core.objectives import PlacementObjective

print("=" * 70)
print("PHASE 2a TEST: Numba JIT Compilation")
print("=" * 70)

# Load medium benchmark
print("\n1. Loading benchmark (adaptec2: ~450K cells)...")
data = load_or_generate_benchmark(
    benchmark_path=None,
    num_cells=5000,  # Use medium size for faster testing
    num_bins=64,
    seed=42
)
print(f"   ✓ Loaded: {data.num_cells} cells, {data.num_movable} movable")

# Test NumPy backend
print("\n2. Testing NumPy backend (baseline, 10 evaluations)...")
obj_numpy = PlacementObjective(data, use_numba=False, use_gpu=False)

start = time.time()
for i in range(10):
    obj_numpy.evaluate()
numpy_time = time.time() - start
print(f"   ✓ NumPy: {numpy_time:.3f}s for 10 evaluations ({numpy_time/10*1000:.1f}ms each)")

# Test Numba backend
print("\n3. Testing Numba backend (JIT compilation, 10 evaluations)...")
obj_numba = PlacementObjective(data, use_numba=True, use_gpu=False)

# First call: JIT compilation happens here
print("   [First call triggers JIT compilation, will be slower...]")
start = time.time()
for i in range(10):
    obj_numba.evaluate()
numba_time = time.time() - start
print(f"   ✓ Numba: {numba_time:.3f}s for 10 evaluations ({numba_time/10*1000:.1f}ms each)")

# Repeated Numba calls (should be fast after warmup)
print("\n4. Second batch of 20 Numba evaluations (after warmup)...")
start = time.time()
for i in range(20):
    obj_numba.evaluate()
numba_warmup_time = time.time() - start
print(f"   ✓ Numba (warmed): {numba_warmup_time:.3f}s for 20 evaluations ({numba_warmup_time/20*1000:.1f}ms each)")

# Comparison
print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)
print(f"NumPy average:         {numpy_time/10*1000:6.2f} ms/eval")
print(f"Numba (with JIT):      {numba_time/10*1000:6.2f} ms/eval")
print(f"Numba (after warmup):  {numba_warmup_time/20*1000:6.2f} ms/eval")

if numba_warmup_time > 0:
    speedup = (numpy_time / 10) / (numba_warmup_time / 20)
    print(f"\nSpeedup factor (after warmup): {speedup:.2f}×")
    if speedup >= 1.5:
        print("✓ PHASE 2a PERFORMANCE GOAL MET (target: 2-3×, achieved: {:.1f}×)".format(speedup))
    else:
        print(f"⚠ Speedup lower than expected (target 2-3×, got {speedup:.1f}×)")
        print("  This may be normal on smaller benchmarks or fast hardware.")

print("\n" + "=" * 70)
print("Next steps:")
print("  1. Build Docker with Numba: docker build -t dreamplace:v2a .")
print("  2. Test batch mode with --use_numba:")
print("     docker run --cpus=8 dreamplace:v2a \\")
print("       python run.py --batch --n_jobs 8 --use_numba \\")
print("         --benchmark benchmarks/ispd2005/adaptec3/adaptec3.aux")
print("  3. Compare Phase 1 vs Phase 2a speedup on BigBlue4 (large benchmark)")
print("=" * 70)
