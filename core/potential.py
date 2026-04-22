"""
Electric potential computation via Poisson equation solver.
Uses FFT-based spectral method (same physics as ePlace/DREAMPlace).

Given a charge density ρ(x,y), solves:
    ∇²φ = -ρ   (Poisson's equation)

Using discrete cosine transform (DCT) for efficient spectral solving.
This replaces the CUDA-based potential operator in DREAMPlace.
"""

import numpy as np
from scipy.fft import dctn, idctn
from .placement import PlacementData


def solve_poisson_dct(density: np.ndarray) -> np.ndarray:
    """
    Solve Poisson equation ∇²φ = -ρ using DCT (Discrete Cosine Transform).

    Uses Neumann boundary conditions (natural for placement problems).
    DCT-II is applied, and eigenvalues of the Laplacian under DCT are used.

    Args:
        density: 2D array of shape (Ny, Nx) — the charge density

    Returns:
        potential: 2D array of shape (Ny, Nx) — the electric potential φ
    """
    Ny, Nx = density.shape

    # Forward DCT of the density
    rho_hat = dctn(density, type=2, norm='ortho')

    # Eigenvalues of the discrete Laplacian under DCT
    # λ_{k,l} = 2(cos(π k/Ny) - 1) + 2(cos(π l/Nx) - 1)
    ky = np.arange(Ny)
    kx = np.arange(Nx)
    KX, KY = np.meshgrid(kx, ky)

    eigenvalues = (
        2.0 * (np.cos(np.pi * KY / Ny) - 1.0) +
        2.0 * (np.cos(np.pi * KX / Nx) - 1.0)
    )

    # Avoid division by zero at (0,0) — set DC component to zero
    eigenvalues[0, 0] = 1.0  # placeholder

    # Solve in spectral space: φ_hat = -ρ_hat / λ
    phi_hat = -rho_hat / eigenvalues
    phi_hat[0, 0] = 0.0  # zero mean potential

    # Inverse DCT
    potential = idctn(phi_hat, type=2, norm='ortho')

    return potential


def solve_poisson_fft(density: np.ndarray) -> np.ndarray:
    """
    Alternative solver using FFT with periodic boundary conditions.
    Treats the density field as periodic.

    Args:
        density: 2D array of shape (Ny, Nx)

    Returns:
        potential: 2D array of shape (Ny, Nx)
    """
    Ny, Nx = density.shape

    # Forward FFT
    rho_hat = np.fft.fft2(density)

    # Frequency grid
    ky = np.fft.fftfreq(Ny) * 2 * np.pi
    kx = np.fft.fftfreq(Nx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_sq = KX**2 + KY**2

    # Avoid division by zero at DC
    k_sq[0, 0] = 1.0

    # Solve: φ_hat = ρ_hat / k²
    phi_hat = rho_hat / k_sq
    phi_hat[0, 0] = 0.0

    # Inverse FFT (take real part)
    potential = np.real(np.fft.ifft2(phi_hat))

    return potential


def compute_potential(data: PlacementData, density: np.ndarray,
                      method: str = "dct") -> np.ndarray:
    """
    Compute electric potential from density map.

    Args:
        data: PlacementData instance
        density: 2D density array
        method: 'dct' (Neumann BC) or 'fft' (periodic BC)

    Returns:
        potential: 2D potential array (same shape as density)
    """
    # Subtract mean to get the "excess" density
    target = np.mean(density)
    rho = density - target

    if method == "dct":
        return solve_poisson_dct(rho)
    elif method == "fft":
        return solve_poisson_fft(rho)
    else:
        raise ValueError(f"Unknown Poisson solver method: {method}")
