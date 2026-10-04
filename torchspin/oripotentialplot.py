"""Plot orientational potential on a sphere.

Port of EasySpin's ``oripotentialplot.m``.  Visualizes the orientational
potential energy or equilibrium population density U(alpha, beta, gamma)
specified as a Wigner expansion.

The potential is parameterized as::

    U(a,b,c) = -kT * sum_i lambda_i * [ D^L_MK(a,b,c) + (-1)^(K-M) * lambda_i* * D^L_{-M,-K}(a,b,c) ]

where ``D^L_MK`` are Wigner D-matrix elements.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from torchspin.angmom import wignerd
from torchspin.rotutils import vec2ang, ang2vec
from torchspin.sphgrid import sphgrid


__all__ = ['oripotentialplot']


def _wignerd_vec(jmm, alpha, beta, gamma):
    """Vectorized Wigner D-matrix element evaluation.

    wignerd() only accepts scalar angles, so we loop over orientations.

    Parameters
    ----------
    jmm : list [J, m1, m2]
    alpha, beta, gamma : ndarray, shape (N,)

    Returns
    -------
    D : ndarray, shape (N,), complex
    """
    alpha = np.atleast_1d(alpha)
    beta = np.atleast_1d(beta)
    gamma = np.atleast_1d(gamma)
    N = len(alpha)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        # wignerd with [J, m1, m2] returns a scalar
        result[k] = complex(wignerd(jmm, float(alpha[k]), float(beta[k]), float(gamma[k])))
    return result


def _lmk_sum(alpha, beta, gamma, lam, Lp, Mp, Kp):
    """Evaluate sum of Wigner D-matrix potential terms.

    Parameters
    ----------
    alpha, beta, gamma : ndarray
        Euler angles (radians).
    lam : ndarray, shape (nTerms,)
        Complex potential coefficients.
    Lp, Mp, Kp : ndarray, shape (nTerms,)
        L, M, K quantum numbers for each term.

    Returns
    -------
    y : ndarray
        Potential energy values (same shape as alpha).
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
    y = np.zeros(len(alpha), dtype=complex)
    for i in range(len(lam)):
        if lam[i] == 0:
            continue
        L, M, K = int(Lp[i]), int(Mp[i]), int(Kp[i])
        if M == 0 and K == 0:
            y += lam[i] * _wignerd_vec([L, 0, 0], alpha, beta, gamma)
        else:
            ph = (-1) ** (K - M)
            y += (lam[i] * _wignerd_vec([L, M, K], alpha, beta, gamma) +
                  ph * np.conj(lam[i]) * _wignerd_vec([L, -M, -K], alpha, beta, gamma))
    return np.real(y)


def oripotentialplot(
    potential: np.ndarray,
    *,
    plot_type: str = 'potential',
    angles: str = 'alpha_beta',
    grid_size: int = 40,
    ax=None,
    cmap: Optional[str] = None,
    kT: float = 2.0,
):
    """Plot orientational potential on a sphere.

    Parameters
    ----------
    potential : ndarray, shape (nTerms, 4)
        Each row is ``[L, M, K, lambda]`` where lambda can be complex.
    plot_type : {'potential', 'population'}
        What to plot: potential energy or Boltzmann population density.
    angles : {'alpha_beta', 'beta_gamma'}
        Which Euler angle pair to display on the sphere.
    grid_size : int
        Number of grid knots for the spherical grid.
    ax : matplotlib Axes3D, optional
        If provided, plot into this axes. Otherwise create a new figure.
    cmap : str, optional
        Colormap name. Default: 'parula'-like for potential, reversed gray
        for population.
    kT : float
        Boltzmann energy scale (arbitrary units). Default: 2.0.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes3D
    values : ndarray
        The plotted values at each grid point.

    Examples
    --------
    >>> import numpy as np
    >>> from torchspin import oripotentialplot
    >>> potential = np.array([[2, 0, 0, 1.5]])
    >>> fig, ax, vals = oripotentialplot(potential)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.tri import Triangulation

    potential = np.asarray(potential, dtype=complex)
    if potential.ndim == 1:
        potential = potential[np.newaxis, :]
    if potential.shape[1] != 4:
        raise ValueError("potential must be an Nx4 array [L, M, K, lambda]")

    # Remove zero-coefficient terms
    mask = potential[:, 3] != 0
    potential = potential[mask]

    Lp = np.real(potential[:, 0]).astype(int)
    Mp = np.real(potential[:, 1]).astype(int)
    Kp = np.real(potential[:, 2]).astype(int)
    lam = potential[:, 3]

    # Validate
    if len(lam) > 0:
        if np.any(Lp < 0):
            raise ValueError("L values must be nonnegative")
        if np.any(np.abs(Kp) > Lp):
            raise ValueError("K values must satisfy |K| <= L")
        if np.any(np.abs(Mp) > Lp):
            raise ValueError("M values must satisfy |M| <= L")

    # Potential and population functions
    def pot_fn(a, b, c):
        return -kT * _lmk_sum(a, b, c, lam, Lp, Mp, Kp)

    def pop_fn(a, b, c):
        return np.exp(-pot_fn(a, b, c) / kT)

    # Build spherical grid
    actual_size = max(grid_size, int(np.max(Lp) * 2) if len(Lp) > 0 else grid_size)
    phi_arr, theta_arr, _, tri_data = sphgrid('C1', actual_size)

    # Get Cartesian vectors
    vecs = ang2vec(phi_arr, theta_arr)  # (3, N)

    # Compute angles from vectors
    phi_grid, theta_grid = vec2ang(vecs)

    if angles == 'alpha_beta':
        alpha = phi_grid if np.ndim(phi_grid) > 0 else np.atleast_1d(phi_grid)
        beta = theta_grid if np.ndim(theta_grid) > 0 else np.atleast_1d(theta_grid)
        gamma_val = np.zeros_like(alpha)
    else:  # beta_gamma
        alpha = np.zeros_like(phi_grid if np.ndim(phi_grid) > 0 else np.atleast_1d(phi_grid))
        beta = theta_grid if np.ndim(theta_grid) > 0 else np.atleast_1d(theta_grid)
        gamma_val = phi_grid if np.ndim(phi_grid) > 0 else np.atleast_1d(phi_grid)

    if plot_type == 'population':
        values = pop_fn(alpha, beta, gamma_val)
        label = 'equilibrium population density'
    else:
        values = pot_fn(alpha, beta, gamma_val) / kT
        label = 'potential energy (kT)'

    # Ensure real
    values = np.real(values)

    # Convert to Cartesian for plotting
    x = np.sin(beta) * np.cos(alpha) if angles == 'alpha_beta' else np.sin(beta) * np.cos(gamma_val)
    y = np.sin(beta) * np.sin(alpha) if angles == 'alpha_beta' else np.sin(beta) * np.sin(gamma_val)
    z = np.cos(beta)

    # Create figure
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Build triangulation — use Delaunay on (phi, theta) for the sphere
    if hasattr(tri_data, 'idx') and tri_data.idx is not None:
        tri_idx = tri_data.idx
        if hasattr(tri_idx, 'numpy'):
            tri_idx = tri_idx.numpy()
        tri_idx = np.asarray(tri_idx, dtype=int)
    else:
        # Fallback: Delaunay triangulation
        from scipy.spatial import Delaunay
        pts_2d = np.column_stack([
            np.asarray(phi_grid).ravel(),
            np.asarray(theta_grid).ravel()
        ])
        tri_idx = Delaunay(pts_2d).simplices

    # Plot
    surf = ax.plot_trisurf(
        np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(z).ravel(),
        triangles=tri_idx,
        cmap=cmap or ('gray_r' if plot_type == 'population' else 'viridis'),
        alpha=0.85,
    )
    surf.set_array(np.asarray(values).ravel())

    # Add axes lines
    v = 1.2
    ax.plot([-v, v], [0, 0], [0, 0], 'k-', linewidth=0.5)
    ax.plot([0, 0], [-v, v], [0, 0], 'k-', linewidth=0.5)
    ax.plot([0, 0], [0, 0], [-v, v], 'k-', linewidth=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    title_angles = r'$(\alpha,\beta,0)$' if angles == 'alpha_beta' else r'$(0,\beta,\gamma)$'
    ax.set_title(f'{label} {title_angles}')

    fig.colorbar(surf, ax=ax, label=label, shrink=0.6)
    ax.set_box_aspect([1, 1, 1])

    return fig, ax, values
