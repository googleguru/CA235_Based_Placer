"""
GUI Visualizer for DREAMPlace MetaOpt.

Displays 4-panel output:
  1. Density Map       — heatmap of cell density per bin
  2. Electric Potential — φ solved via Poisson equation
  3. Electric Field     — E = -∇φ magnitude
  4. Cell Grid          — cells placed on discretized grid

Also shows algorithm selection, progress bars, and metrics.
Uses Matplotlib with embedded Tkinter for interactive GUI.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.placement import PlacementData
from core.density import compute_density_map_fast, compute_target_density
from core.potential import compute_potential
from core.field import compute_field, compute_field_magnitude
from core.wirelength import compute_hpwl


# ═══════════════════════════════════════════════════════════════════════════════
# Custom color maps for a premium look
# ═══════════════════════════════════════════════════════════════════════════════

DENSITY_CMAP = LinearSegmentedColormap.from_list("density", [
    "#0a0a2e", "#1a1a6e", "#2828a8", "#3c78d8",
    "#00c853", "#ffeb3b", "#ff9800", "#f44336", "#b71c1c"
])

POTENTIAL_CMAP = LinearSegmentedColormap.from_list("potential", [
    "#001f3f", "#003366", "#0056a8", "#0088cc",
    "#66bbee", "#aaddff", "#ffffff",
    "#ffccaa", "#ff8866", "#cc3300", "#880000"
])

FIELD_CMAP = LinearSegmentedColormap.from_list("field", [
    "#000033", "#001166", "#0033cc", "#0066ff",
    "#33ccff", "#66ffcc", "#99ff66",
    "#ccff33", "#ffcc00", "#ff6600", "#cc0000"
])

GRID_CMAP = LinearSegmentedColormap.from_list("grid", [
    "#0d1117", "#161b22", "#1a3a4a", "#1e6050",
    "#2ea043", "#57d364", "#a6f0a6"
])


# ═══════════════════════════════════════════════════════════════════════════════
# Main GUI Application
# ═══════════════════════════════════════════════════════════════════════════════

class PlacementVisualizer:
    """
    Main GUI window showing 4-panel placement analysis.
    """

    def __init__(self, root: tk.Tk = None):
        self.root = root or tk.Tk()
        self.root.title("DREAMPlace MetaOpt - VLSI Placement Visualizer")
        self.root.configure(bg="#0d1117")

        # State
        self.data: PlacementData = None
        self.results = {}  # algorithm_name → result dict
        self.running = False
        self.current_algorithm = None

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        """Configure dark theme styles."""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("Dark.TFrame", background="#0d1117")
        style.configure("Dark.TLabel", background="#0d1117", foreground="#c9d1d9",
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#0d1117", foreground="#58a6ff",
                        font=("Segoe UI", 14, "bold"))
        style.configure("Metric.TLabel", background="#161b22", foreground="#7ee787",
                        font=("Consolas", 11))
        style.configure("Dark.TButton", background="#21262d", foreground="#c9d1d9",
                        font=("Segoe UI", 10, "bold"))
        style.map("Dark.TButton",
                 background=[("active", "#30363d"), ("pressed", "#484f58")])
        style.configure("Accent.TButton", background="#238636", foreground="#ffffff",
                        font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton",
                 background=[("active", "#2ea043"), ("pressed", "#1a7f37")])
        style.configure("Dark.TLabelframe", background="#0d1117", foreground="#58a6ff",
                        font=("Segoe UI", 10, "bold"))
        style.configure("Dark.TLabelframe.Label", background="#0d1117", foreground="#58a6ff")
        style.configure("Dark.TRadiobutton", background="#0d1117", foreground="#c9d1d9",
                        font=("Segoe UI", 10))
        style.map("Dark.TRadiobutton",
                 background=[("active", "#161b22")])
        style.configure("Green.Horizontal.TProgressbar",
                        background="#2ea043", troughcolor="#21262d")

    def _build_ui(self):
        """Build the main UI layout."""
        # ── Top bar ──────────────────────────────────────────────────────────
        top_frame = ttk.Frame(self.root, style="Dark.TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(top_frame, text="🛣️ DREAMPlace MetaOpt",
                 style="Title.TLabel").pack(side=tk.LEFT, padx=5)

        ttk.Label(top_frame, text="No Deep Learning - Pure Metaheuristic Placement",
                 style="Dark.TLabel").pack(side=tk.LEFT, padx=20)

        # ── Control panel ────────────────────────────────────────────────────
        ctrl_frame = ttk.LabelFrame(self.root, text="⚙️  Algorithm Control",
                                    style="Dark.TLabelframe")
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        inner = ttk.Frame(ctrl_frame, style="Dark.TFrame")
        inner.pack(fill=tk.X, padx=10, pady=5)

        # Algorithm selection
        self.algo_var = tk.StringVar(value="all")

        algos = [
            ("🔹 Hybrid SBO", "sbo"),
            ("🔸 PSO + SBO", "pso"),
            ("🔻 Cellular Automata CA235", "ca235"),
            ("🔷 Run All Three", "all"),
        ]
        for text, val in algos:
            ttk.Radiobutton(inner, text=text, variable=self.algo_var,
                           value=val, style="Dark.TRadiobutton").pack(
                               side=tk.LEFT, padx=10)

        # Run button
        self.run_btn = ttk.Button(inner, text="▶  RUN", style="Accent.TButton",
                                  command=self._on_run)
        self.run_btn.pack(side=tk.RIGHT, padx=10)

        # Progress bar
        self.progress = ttk.Progressbar(ctrl_frame, style="Green.Horizontal.TProgressbar",
                                         mode='indeterminate', length=300)
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Status label
        self.status_var = tk.StringVar(value="Ready. Click RUN to start placement.")
        ttk.Label(ctrl_frame, textvariable=self.status_var,
                 style="Dark.TLabel").pack(padx=10, pady=(0, 5))

        # ── Main visualization area ──────────────────────────────────────────
        viz_frame = ttk.Frame(self.root, style="Dark.TFrame")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(16, 10), dpi=100, facecolor='#0d1117')
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = ttk.Frame(viz_frame, style="Dark.TFrame")
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # ── Metrics panel ────────────────────────────────────────────────────
        metrics_frame = ttk.LabelFrame(self.root, text="📊  Metrics",
                                       style="Dark.TLabelframe")
        metrics_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.metrics_text = tk.Text(metrics_frame, height=4, bg="#161b22", fg="#7ee787",
                                    font=("Consolas", 10), relief=tk.FLAT,
                                    insertbackground="#7ee787")
        self.metrics_text.pack(fill=tk.X, padx=10, pady=5)
        self.metrics_text.insert(tk.END, "Metrics will appear here after running an algorithm.\n")
        self.metrics_text.config(state=tk.DISABLED)

        # Initial empty plot
        self._draw_empty()

    def _draw_empty(self):
        """Draw screenshot-style placeholder panels (1x4 table layout)."""
        self.fig.clear()
        self.fig.patch.set_facecolor('#020817')
        gs = gridspec.GridSpec(1, 4, figure=self.fig, left=0.03, right=0.985,
                               top=0.90, bottom=0.08, wspace=0.12)

        titles = ["Bigblue4", "Density Map", "Electric Potential", "Electric Field"]
        for i, title in enumerate(titles):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_facecolor('#0a1020')
            ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#2b3448')
                spine.set_linewidth(1.0)
            ax.text(0.5, 0.5, "Waiting for run...", transform=ax.transAxes,
                    ha='center', va='center', color='#7f8ca3', fontsize=12)

        self.canvas.draw()

    def _make_bigblue_overlay(self, density, Ex, Ey):
        """Build a BigBlue-like panel (red blocks + blue electric traces)."""
        h, w = density.shape
        occupancy = np.zeros((h, w), dtype=np.float64)

        mov_idx = self.data.get_movable_indices()
        if len(mov_idx) > 0:
            cx = self.data.cell_x[mov_idx] + self.data.cell_w[mov_idx] / 2.0
            cy = self.data.cell_y[mov_idx] + self.data.cell_h[mov_idx] / 2.0
            ix = np.clip(((cx - self.data.die_xl) / max(self.data.die_width, 1e-9) * w).astype(int), 0, w - 1)
            iy = np.clip(((cy - self.data.die_yl) / max(self.data.die_height, 1e-9) * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 1.0)

        fixed_idx = np.where(self.data.fixed_mask)[0]
        if len(fixed_idx) > 0:
            fx = self.data.cell_x[fixed_idx] + self.data.cell_w[fixed_idx] / 2.0
            fy = self.data.cell_y[fixed_idx] + self.data.cell_h[fixed_idx] / 2.0
            ix = np.clip(((fx - self.data.die_xl) / max(self.data.die_width, 1e-9) * w).astype(int), 0, w - 1)
            iy = np.clip(((fy - self.data.die_yl) / max(self.data.die_height, 1e-9) * h).astype(int), 0, h - 1)
            np.add.at(occupancy, (iy, ix), 3.0)

        occ_max = float(np.max(occupancy))
        if occ_max > 0:
            occupancy = occupancy / occ_max

        field_mag = np.sqrt(Ex ** 2 + Ey ** 2)
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

    def set_data(self, data: PlacementData):
        """Set the placement data."""
        self.data = data
        self.status_var.set(f"Loaded: {data.name} | {data.num_movable:,} movable cells | "
                           f"{data.num_nets:,} nets")

    def _on_run(self):
        """Handle Run button click."""
        if self.data is None:
            messagebox.showwarning("No Data", "Please load a benchmark first.")
            return

        if self.running:
            return

        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.progress.start(10)

        algo = self.algo_var.get()
        thread = threading.Thread(target=self._run_algorithms, args=(algo,), daemon=True)
        thread.start()

    def _run_algorithms(self, algo_choice: str):
        """Run selected algorithm(s) in background thread."""
        try:
            from core.objectives import PlacementObjective
            from algorithms.hybrid_sbo import HybridSBO
            from algorithms.pso_sbo import PSOWithSBO
            from algorithms.cellular_automata import CellularAutomataCA235

            algorithms_to_run = []
            if algo_choice in ("sbo", "all"):
                algorithms_to_run.append("sbo")
            if algo_choice in ("pso", "all"):
                algorithms_to_run.append("pso")
            if algo_choice in ("ca235", "all"):
                algorithms_to_run.append("ca235")

            for algo_name in algorithms_to_run:
                # Reset positions for each algorithm
                self.data.random_placement(seed=42)

                objective = PlacementObjective(self.data)

                self.root.after(0, lambda n=algo_name: self.status_var.set(
                    f"Running {n.upper()}..."))

                if algo_name == "sbo":
                    algo = HybridSBO(self.data, objective, max_iterations=800,
                                    num_clusters=min(50, self.data.num_movable // 10))
                    result = algo.run()
                elif algo_name == "pso":
                    algo = PSOWithSBO(self.data, objective, max_iterations=800,
                                    num_clusters=min(50, self.data.num_movable // 10),
                                    num_particles=20)
                    result = algo.run()
                elif algo_name == "ca235":
                    algo = CellularAutomataCA235(self.data, objective, max_iterations=800)
                    result = algo.run()

                self.results[algo_name] = result

                # Update visualization after each algorithm
                self.root.after(0, lambda r=result: self._update_plots(r))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            import traceback
            traceback.print_exc()
        finally:
            self.root.after(0, self._on_run_complete)

    def _on_run_complete(self):
        """Called when algorithm run finishes."""
        self.running = False
        self.run_btn.config(state=tk.NORMAL)
        self.progress.stop()
        self.status_var.set("✅ Placement complete! See results below.")

    def _update_plots(self, result: dict):
        """Update all 4 panels with screenshot-style output."""
        self.fig.clear()
        self.fig.patch.set_facecolor('#020817')

        density = result.get('density_map', compute_density_map_fast(self.data))
        potential = compute_potential(self.data, density, method='dct')
        Ex, Ey = compute_field(potential, self.data)
        field_mag = compute_field_magnitude(Ex, Ey)

        gs = gridspec.GridSpec(1, 4, figure=self.fig, left=0.03, right=0.985,
                               top=0.90, bottom=0.08, wspace=0.12)

        # Panel 1: Bigblue4-like composition
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.imshow(self._make_bigblue_overlay(density, Ex, Ey), origin='lower', interpolation='nearest')
        ax1.set_title("Bigblue4", color='white', fontsize=13, fontweight='bold', pad=10)

        iter_count = result.get('iterations', len(result.get('history', [])))
        if iter_count:
            ax1.text(0.97, 0.03, f"Iter: {iter_count}", transform=ax1.transAxes,
                     ha='right', va='bottom', color='#2f6df6', fontsize=11, fontweight='bold')

        # Panel 2: Density map (grayscale)
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.imshow(density, cmap='Greys', origin='lower', interpolation='nearest')
        ax2.set_title("Density Map", color='white', fontsize=13, fontweight='bold', pad=10)

        # Panel 3: Electric potential (grayscale)
        ax3 = self.fig.add_subplot(gs[0, 2])
        ax3.imshow(potential, cmap='gray', origin='lower', interpolation='nearest')
        ax3.set_title("Electric Potential", color='white', fontsize=13, fontweight='bold', pad=10)

        # Panel 4: Electric field magnitude (grayscale)
        ax4 = self.fig.add_subplot(gs[0, 3])
        ax4.imshow(field_mag, cmap='gray', origin='lower', interpolation='nearest')
        ax4.set_title("Electric Field", color='white', fontsize=13, fontweight='bold', pad=10)

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#2b3448')
                spine.set_linewidth(1.0)

        self.canvas.draw()

        # Update metrics
        self._update_metrics(result)

    def _draw_cell_grid(self, ax, algo_name: str):
        """Draw cells placed on a grid."""
        ax.set_facecolor('#0d1117')

        nbx = self.data.num_bins_x
        nby = self.data.num_bins_y
        bw = self.data.bin_width
        bh = self.data.bin_height

        # Draw grid lines
        for i in range(nbx + 1):
            x = self.data.die_xl + i * bw
            ax.axvline(x, color='#21262d', linewidth=0.3, alpha=0.5)
        for j in range(nby + 1):
            y = self.data.die_yl + j * bh
            ax.axhline(y, color='#21262d', linewidth=0.3, alpha=0.5)

        # Plot cells as small dots
        idx = self.data.get_movable_indices()
        max_points = min(5000, len(idx))  # Limit points for performance
        if len(idx) > max_points:
            sample = np.random.choice(len(idx), max_points, replace=False)
            plot_idx = idx[sample]
        else:
            plot_idx = idx

        cx = self.data.cell_x[plot_idx] + self.data.cell_w[plot_idx] / 2.0
        cy = self.data.cell_y[plot_idx] + self.data.cell_h[plot_idx] / 2.0

        ax.scatter(cx, cy, s=0.5, c='#57d364', alpha=0.3, edgecolors='none')

        # Fixed cells in different color
        fixed_idx = np.where(self.data.fixed_mask)[0]
        if len(fixed_idx) > 0:
            fx = self.data.cell_x[fixed_idx] + self.data.cell_w[fixed_idx] / 2.0
            fy = self.data.cell_y[fixed_idx] + self.data.cell_h[fixed_idx] / 2.0
            ax.scatter(fx, fy, s=3, c='#f78166', alpha=0.7, edgecolors='none',
                      marker='s', label='Fixed')

        # Zoom in if movable cells don't span the full die (e.g. low density clusters)
        mov_idx = self.data.get_movable_indices()
        if len(mov_idx) > 0:
            min_x_mov = self.data.cell_x[mov_idx].min()
            max_x_mov = (self.data.cell_x[mov_idx] + self.data.cell_w[mov_idx]).max()
            min_y_mov = self.data.cell_y[mov_idx].min()
            max_y_mov = (self.data.cell_y[mov_idx] + self.data.cell_h[mov_idx]).max()
            
            bb_w = max_x_mov - min_x_mov
            bb_h = max_y_mov - min_y_mov
            
            if bb_w < 0.85 * self.data.die_width or bb_h < 0.85 * self.data.die_height:
                pad_x = max(bb_w * 0.15, 100)
                pad_y = max(bb_h * 0.15, 100)
                ax.set_xlim(max(self.data.die_xl, min_x_mov - pad_x), min(self.data.die_xh, max_x_mov + pad_x))
                ax.set_ylim(max(self.data.die_yl, min_y_mov - pad_y), min(self.data.die_yh, max_y_mov + pad_y))
            else:
                ax.set_xlim(self.data.die_xl, self.data.die_xh)
                ax.set_ylim(self.data.die_yl, self.data.die_yh)
        else:
            ax.set_xlim(self.data.die_xl, self.data.die_xh)
            ax.set_ylim(self.data.die_yl, self.data.die_yh)
        ax.set_title(f"Cell Grid Placement - {algo_name}", color='#58a6ff',
                    fontsize=11, fontweight='bold', pad=8)
        self._style_axis(ax)

    def _style_axis(self, ax):
        """Apply dark theme to an axis."""
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#484f58', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    def _update_metrics(self, result: dict):
        """Update the metrics text panel."""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)

        algo = result.get('algorithm', '?')
        hpwl = result.get('hpwl', 0)
        overflow = result.get('density_overflow', 0)
        runtime = result.get('runtime', 0)

        lines = [
            f"Algorithm: {algo}",
            f"  HPWL:           {hpwl:,.0f}",
            f"  Density Overflow: {overflow:.6f}",
            f"  Runtime:        {runtime:.2f}s",
            f"  Cells:          {self.data.num_movable:,} movable / {self.data.num_fixed:,} fixed",
            f"  Grid:           {self.data.num_bins_x}×{self.data.num_bins_y} bins",
        ]

        # If we have results for multiple algorithms, show comparison
        if len(self.results) > 1:
            lines.append("")
            lines.append("--- Algorithm Comparison ---")
            lines.append(f"  {'Algorithm':<20s} {'HPWL':>12s} {'Overflow':>12s} {'Time':>8s}")
            for name, res in self.results.items():
                lines.append(f"  {res.get('algorithm', name):<20s} "
                           f"{res.get('hpwl', 0):>12,.0f} "
                           f"{res.get('density_overflow', 0):>12.6f} "
                           f"{res.get('runtime', 0):>7.1f}s")

        self.metrics_text.insert(tk.END, "\n".join(lines))
        self.metrics_text.config(state=tk.DISABLED)

    def _save_results(self, filename="results.png"):
        """Save the current visualization as a high-quality PNG file."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Save Results", f"Results saved as {filename}")

    def run(self):
        """Start the GUI event loop."""
        # Set minimum window size
        self.root.minsize(1200, 800)
        # Center on screen
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

        self.root.mainloop()


def launch_gui(data: PlacementData = None):
    """
    Launch the placement visualizer GUI.

    Args:
        data: Optional pre-loaded PlacementData. If None, a synthetic benchmark is generated.
    """
    if data is None:
        from core.benchmark_parser import generate_synthetic_benchmark
        data = generate_synthetic_benchmark(num_cells=3000, num_bins=64)

    root = tk.Tk()
    app = PlacementVisualizer(root)
    app.set_data(data)
    app.run()


if __name__ == "__main__":
    launch_gui()
