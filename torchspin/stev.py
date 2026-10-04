"""Extended Stevens spin operator matrices for torchspin.

Port of EasySpin's ``stev.m``. Constructs extended Stevens operators O_k^q
of rank k and component q for a given spin quantum number S.

Stevens operators are tesseral (as opposed to spherical) tensor operators
and are all Hermitian. They are used to describe crystal-field and zero-field
splitting interactions in high-spin systems (S > 1/2).

Convention
----------
* q < 0 → sine tesseral components O_k^|q|(s)
* q > 0 → cosine tesseral components O_k^|q|(c)
* q = 0 → axial component O_k^0

Ranks from 0 to 12 are supported. Most common in EPR are k = 2, 4, 6.

Usage
-----
::

    from torchspin.stev import stev
    import torch

    # O_4^2(c) for spin S=5/2
    Op = stev(2.5, k=4, q=2)

    # O_6^-5(s) for the second spin in a two-spin system
    Op = stev([1.5, 1.5], k=6, q=-5, iSpin=2)

    # Sparse output
    Op_sparse = stev(2.5, k=4, q=2, sparse=True)

References
----------
* I.D. Ryabov, J. Magn. Reson. 140, 141-145 (1999)
  https://doi.org/10.1006/jmre.1999.1783
* C. Rudowicz, C.Y. Chung, J. Phys.: Condens. Matter 16, 1-23 (2004)
* Altshuler/Kozyrev, Electron Paramagnetic Resonance in Compounds of
  Transition Elements, Wiley, 2nd edn., (1974), Appendix V, p. 512
* Abragam/Bleaney, Electron Paramagnetic Resonance of Transition Ions,
  Dover (1986), Table 16, p. 863
"""
from __future__ import annotations

from typing import Optional

import torch

from torchspin.spinops import sop


# ---------------------------------------------------------------------------
# Normalization prefactors F[k, q] for Stevens operators
# Computed using Ryabov, J. Magn. Reson. 140, 141-145 (1999)
# These are the values from EasySpin stev.m
# ---------------------------------------------------------------------------
_F = {
    0: [1],
    1: [2, 1],
    2: [4, 2, 1],
    3: [24, 6, 6, 1],
    4: [48, 24, 8, 4, 1],
    5: [480, 240, 240, 10, 10, 1],
    6: [2880, 1440, 360, 60, 12, 6, 1],
    7: [40320, 5040, 1680, 168, 168, 14, 14, 1],
    8: [80640, 40320, 40320, 6720, 672, 336, 16, 8, 1],
    9: [1451520, 725700, 725700, 60480, 60480, 864, 288, 18, 18, 1],
    10: [14515200, 7257600, 1209600, 604800, 86400, 2880, 360, 180, 20, 10, 1],
    11: [319334400, 79833600, 79833600, 13305600, 2661120, 23760, 7920, 1320, 1320, 22, 22, 1],
    12: [1916006400, 958003200, 958003200, 31933440, 3991680, 1995840, 31680, 15840, 1584, 264, 24, 12, 1],
}

_KMAX = 12  # Maximum supported rank


def _Fval(k: int, q: int) -> float:
    """Return prefactor F[k, q] for Stevens operator normalization."""
    return float(_F[k][q])


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def stev(
    Spins: float | list[float],
    k: int,
    q: int,
    iSpin: Optional[int] = None,
    sparse: bool = False,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Construct extended Stevens operator O_k^q.

    Parameters
    ----------
    Spins : float or list of float
        Spin quantum number(s). If a list, specifies a multi-spin system.
    k : int
        Rank of the Stevens operator. Must satisfy 0 ≤ k ≤ 2*S and k ≤ 12.
        Common values are k=2, 4, 6.
    q : int
        Component index. Must satisfy -k ≤ q ≤ k.
        * q > 0: cosine tesseral component O_k^q(c)
        * q < 0: sine tesseral component O_k^|q|(s)
        * q = 0: axial component O_k^0
    iSpin : int, optional
        Index (1-based) of the spin in a multi-spin system for which to
        compute the operator. Required if ``Spins`` is a list with more
        than one element. Default is 1 for single-spin systems.
    sparse : bool, optional
        If True, return a sparse tensor (COO format). Default is False.
    dtype : torch.dtype, optional
        Data type for the output tensor. Default is ``torch.complex128``.

    Returns
    -------
    Op : torch.Tensor
        Stevens operator matrix of shape ``(dim, dim)`` where
        ``dim = prod([2*S_i+1 for S_i in Spins])``.

    Raises
    ------
    ValueError
        If any of k, q, iSpin are out of valid range.

    Examples
    --------
    >>> from torchspin.stev import stev
    >>> # Axial ZFS operator O_2^0 for S=1
    >>> Op = stev(1.0, k=2, q=0)
    >>> Op.shape
    torch.Size([3, 3])

    >>> # Rhombic ZFS operator O_2^2(c) for S=1
    >>> Op = stev(1.0, k=2, q=2)

    >>> # High-order operator O_4^0 for Mn(II) with S=5/2
    >>> Op = stev(2.5, k=4, q=0)
    >>> Op.shape
    torch.Size([6, 6])

    >>> # Multi-spin: O_2^0 for second spin in [S=1/2, S=1] system
    >>> Op = stev([0.5, 1.0], k=2, q=0, iSpin=2)
    >>> Op.shape
    torch.Size([6, 6])

    Notes
    -----
    The extended Stevens operators are constructed using Racah's commutation
    rule starting from J_+^k, then forming cosine and sine tesseral operators
    from spherical tensor components. See Ryabov (1999) for the algorithm.

    The operators are Hermitian and satisfy standard commutation relations
    for tesseral tensor operators.
    """
    # Parse Spins input
    if isinstance(Spins, (int, float)):
        Spins = [float(Spins)]
    elif isinstance(Spins, list):
        Spins = [float(s) for s in Spins]
    else:
        raise TypeError(f"Spins must be float or list[float], not {type(Spins)}")

    # Validate spins are positive half-integers
    for s in Spins:
        if s < 0 or (2 * s) % 1 != 0:
            raise ValueError("All spins must be non-negative multiples of 1/2")

    # Handle iSpin default
    if iSpin is None:
        if len(Spins) > 1:
            raise ValueError(
                "For multi-spin system, iSpin must be specified (1-based index)"
            )
        iSpin = 1

    # Validate iSpin
    if not isinstance(iSpin, int) or iSpin < 1 or iSpin > len(Spins):
        raise ValueError(
            f"iSpin={iSpin} out of range. Must be between 1 and {len(Spins)}"
        )

    S = Spins[iSpin - 1]  # Convert to 0-based indexing

    # Validate k
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if k > _KMAX:
        raise ValueError(f"k={k} too large. Maximum supported k is {_KMAX}")
    if k > 2 * S:
        raise ValueError(
            f"k={k} exceeds 2*S={2*S} for S={S}. "
            "Stevens operators require k ≤ 2*S"
        )

    # Validate q
    if not isinstance(q, int) or abs(q) > k:
        raise ValueError(f"q must be an integer with |q| ≤ k. Got q={q}, k={k}")

    # Build spherical tensor operator T^k_|q| using Racah's commutation rule
    # Start with T^k_k = J_+^k, then apply J_- repeatedly
    # (Ryabov Eq. [1])

    # Get J_+ and J_- operators
    # sop numeric syntax: component 4=+, 5=-
    Jp = sop(Spins, [[iSpin, 4]], dtype=dtype).to_sparse()  # J_+
    Jm = sop(Spins, [[iSpin, 5]], dtype=dtype).to_sparse()  # J_-

    # Special case: k=0 gives identity
    if k == 0:
        dim =  int(torch.prod(torch.tensor([2*s+1 for s in Spins])))
        T = torch.eye(dim, dtype=dtype).to_sparse()
    else:
        # Start with T = J_+^k
        T = Jp.clone()
        for _ in range(k - 1):
            T = T @ Jp

    # Apply lowering: T^k_q = (J_- T - T J_-)^(k-q)
    for _ in range(k - abs(q)):
        T = Jm @ T - T @ Jm

    # Compute Stevens prefactor (Ryabov Eq. [22])
    # alpha factor depends on parity of k and q
    if k % 2 == 1:  # odd k
        alpha = 1.0
    else:  # even k
        if abs(q) % 2 == 1:  # odd q
            alpha = 0.5
        else:  # even q
            alpha = 1.0

    c = alpha / _Fval(k, abs(q))

    # Retain sign of original normalization constant
    c *= (-1) ** (k - q)

    # Construct cosine or sine tesseral operator (Ryabov Eq. [21])
    if q >= 0:
        # Cosine tesseral: (T + T†) / 2
        Op = c / 2 * (T + T.H)
    else:
        # Sine tesseral: (T - T†) / (2i)
        Op = c / (2j) * (T - T.H)

    # Clean small numerical errors (threshold from MATLAB)
    Op_dense = Op.to_dense() if Op.is_sparse else Op
    Op_dense[Op_dense.abs() < 1e-14] = 0

    # Return sparse or dense
    if sparse:
        return Op_dense.to_sparse()
    else:
        return Op_dense


# ---------------------------------------------------------------------------
# Convenience: common Stevens operators for EPR
# ---------------------------------------------------------------------------
def O20(S: float, **kwargs) -> torch.Tensor:
    """Axial ZFS operator O_2^0 (proportional to 3*Sz^2 - S(S+1))."""
    return stev(S, k=2, q=0, **kwargs)


def O22c(S: float, **kwargs) -> torch.Tensor:
    """Rhombic ZFS operator O_2^2(c) (proportional to Sx^2 - Sy^2)."""
    return stev(S, k=2, q=2, **kwargs)


def O40(S: float, **kwargs) -> torch.Tensor:
    """Fourth-order axial operator O_4^0."""
    return stev(S, k=4, q=0, **kwargs)


def O44c(S: float, **kwargs) -> torch.Tensor:
    """Fourth-order rhombic operator O_4^4(c)."""
    return stev(S, k=4, q=4, **kwargs)


def O60(S: float, **kwargs) -> torch.Tensor:
    """Sixth-order axial operator O_6^0."""
    return stev(S, k=6, q=0, **kwargs)
