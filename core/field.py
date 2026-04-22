"""
Electric field computation from potential.
The electric field is the negative gradient of the potential:
    E = -∇φ  →  Ex = -∂φ/∂x,  Ey = -∂φ/∂y

In the ePlace methodology, the electric field provides the
"force" that pushes cells away from high-density regions.
"""

import numpy as np
from .placement import PlacementData


def compute_field(potential: np.ndarray, data: PlacementData) -> tuple:
    """
    Compute electric field from the potential map.

    Uses central finite differences scaled to physical coordinates.

    Args:
        potential: 2D potential array (Ny, Nx)
        data: PlacementData for coordinate scaling

    Returns:
        (Ex, Ey): Tuple of 2D arrays, each shape (Ny, Nx)
                  Ex = -∂φ/∂x (horizontal field)
                  Ey = -∂φ/∂y (vertical field)
    """
    Ny, Nx = potential.shape

    # Physical spacing between bins
    dx = data.bin_width if data.bin_width > 0 else 1.0
    dy = data.bin_height if data.bin_height > 0 else 1.0

    # Central differences with zero-padding at boundaries
    # ∂φ/∂x: gradient along columns (axis=1)
    Ex = np.zeros_like(potential)
    Ex[:, 1:-1] = -(potential[:, 2:] - potential[:, :-2]) / (2.0 * dx)
    Ex[:, 0] = -(potential[:, 1] - potential[:, 0]) / dx
    Ex[:, -1] = -(potential[:, -1] - potential[:, -2]) / dx

    # ∂φ/∂y: gradient along rows (axis=0)
    Ey = np.zeros_like(potential)
    Ey[1:-1, :] = -(potential[2:, :] - potential[:-2, :]) / (2.0 * dy)
    Ey[0, :] = -(potential[1, :] - potential[0, :]) / dy
    Ey[-1, :] = -(potential[-1, :] - potential[-2, :]) / dy

    return Ex, Ey


def compute_field_magnitude(Ex: np.ndarray, Ey: np.ndarray) -> np.ndarray:
    """
    Compute electric field magnitude: |E| = sqrt(Ex² + Ey²)

    Returns:
        magnitude: 2D array (Ny, Nx)
    """
    return np.sqrt(Ex**2 + Ey**2)


def compute_field_spectral(potential: np.ndarray) -> tuple:
    """
    Compute electric field using spectral differentiation (FFT).
    More accurate than finite differences for smooth fields.

    Returns:
        (Ex, Ey): Tuple of 2D arrays
    """
    Ny, Nx = potential.shape

    # FFT of potential
    phi_hat = np.fft.fft2(potential)

    # Frequency grids
    ky = np.fft.fftfreq(Ny) * 2 * np.pi
    kx = np.fft.fftfreq(Nx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    # Spectral derivative: ∂/∂x → multiply by i*kx
    Ex = -np.real(np.fft.ifft2(1j * KX * phi_hat))
    Ey = -np.real(np.fft.ifft2(1j * KY * phi_hat))

    return Ex, Ey
