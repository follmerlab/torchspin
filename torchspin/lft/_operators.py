"""Parameter-independent d^n operators as torch constants.

The d^n Hamiltonian is linear in every parameter over fixed matrices
(:mod:`torchspin.lft._engines.dn`), so only the parameter-dependent assembly runs on the
autograd graph.  The NumPy matrices are converted once per (n, device) and cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from ._engines.dn import U_REAL, term_projectors
from ._engines.dn import operators as np_operators

__all__ = ["DnOperators", "dn_operators"]


@dataclass(frozen=True, eq=False)
class DnOperators:
    """Torch constants of d^n (nd = C(10, n) determinants; complex128 unless noted).

    ``coulomb``   (3, nd, nd) float64: coefficients of F^0, F^2, F^4
    ``racah``     (3, nd, nd) float64: coefficients of Racah A, B, C
    ``lf_real``   (5, 5, nd, nd): E_real[a, b], H_LF = sum_ab V_real[a, b] E_real[a, b]
                  (real orbitals x2-y2, xy, xz, yz, z2)
    ``so``        (nd, nd) float64: sum_i l_i . s_i
    ``L``, ``S``  (3, nd, nd): total orbital and spin angular momentum
    ``L2``, ``S2`` (nd, nd) float64
    ``term_projectors`` (nterms, nd, nd) float64, DDN term order
    """

    n: int
    coulomb: torch.Tensor
    racah: torch.Tensor
    lf_real: torch.Tensor
    so: torch.Tensor
    L: torch.Tensor
    S: torch.Tensor
    L2: torch.Tensor
    S2: torch.Tensor
    term_projectors: torch.Tensor

    @property
    def size(self) -> int:
        return self.so.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.so.device


@lru_cache(maxsize=None)
def _dn_operators(n: int, device: str) -> DnOperators:
    ops = np_operators(n)
    c0, c2, c4 = ops.coulomb
    racah = np.array([c0, 49.0 * c2, 7.0 / 5.0 * c0 + 7.0 * c2 + 63.0 / 5.0 * c4])
    u = U_REAL
    lf_real = np.einsum("ma,nb,mnij->abij", u, u.conj(), ops.excitation)
    _, projs = term_projectors(n)
    f64 = lambda a: torch.as_tensor(np.array(a), dtype=torch.float64, device=device)  # noqa: E731
    c128 = lambda a: torch.as_tensor(np.array(a), dtype=torch.complex128, device=device)  # noqa: E731
    return DnOperators(n=n, coulomb=f64(ops.coulomb), racah=f64(racah), lf_real=c128(lf_real), so=f64(ops.so),
                       L=c128(ops.L), S=c128(ops.S), L2=f64(ops.L2), S2=f64(ops.S2), term_projectors=f64(projs))


def dn_operators(n: int, device: str | torch.device = "cpu") -> DnOperators:
    """Cached torch operators of d^n on ``device``."""
    if int(n) != n or not 0 <= n <= 10:
        raise ValueError("n must be an integer 0..10")
    return _dn_operators(int(n), str(torch.device(device)))
