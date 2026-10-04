"""Multi-spin operator matrices for torchspin.

``sop`` constructs spin operator matrices for single or multi-spin systems
using the same basis ordering as EasySpin: m = S, S-1, ..., -S (descending).

For a multi-spin system the full Hilbert space is the tensor product of the
individual spin spaces.  An operator for spin *i* with component *c* is built
as the Kronecker product of single-spin operators/identities in each position.

Example::

    from torchspin.spinops import sop

    # Single spin-1/2 Sx
    Sx = sop([0.5], 'x')

    # Two-spin system: Sx ⊗ I  (operator on first spin only)
    SxI = sop([0.5, 0.5], 'xe')

    # Two-spin system: I ⊗ Sy  (operator on second spin only)
    ISy = sop([0.5, 0.5], 'ey')

    # Two-spin system using numeric spec: Sx on spin 1
    SxI = sop([0.5, 0.5], [[1, 1]])   # spin-index 1, component x (=1)

    # Product operator Sx(1) * Sz(2)
    SxSz = sop([0.5, 0.5], [[1, 1], [2, 3]])
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Component index mapping (matches EasySpin numeric spec)
# 0=e (identity), 1=x, 2=y, 3=z, 4=+, 5=-
# ---------------------------------------------------------------------------
_CHAR_TO_IDX: dict[str, int] = {
    'e': 0, 'x': 1, 'y': 2, 'z': 3, '+': 4, '-': 5,
}


def _single_sop(S: float, comp: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Return a single-spin operator for spin S and component index comp.

    Parameters
    ----------
    S:
        Spin quantum number (0, 0.5, 1, 1.5, ...).
    comp:
        Component: 0=identity, 1=Sx, 2=Sy, 3=Sz, 4=S+, 5=S-.
    dtype:
        Complex dtype (torch.complex64 or torch.complex128).
    dtype:
        PyTorch device string.
    """
    S = float(S)
    n = int(round(2 * S + 1))

    if comp == 0:  # identity
        return torch.eye(n, dtype=dtype, device=device)

    # Build in float64 for numerical stability, cast at the end
    fdtype = torch.float64

    i = torch.arange(n, dtype=fdtype, device=device)
    m = S - i  # m values: S, S-1, ..., -S

    if n > 1:
        # S+ (upper off-diagonal at position +1):  <m+1|S+|m> = sqrt(S(S+1)-m(m+1))
        m_sub = m[1:]  # m values for column index of S+: m = S-1, ..., -S
        vals_p = torch.sqrt((S * (S + 1) - m_sub * (m_sub + 1)).clamp(min=0.0))
        Sp = torch.diag(vals_p, 1)  # (n,n) — row i, col i+1

        # S- (lower off-diagonal at position -1):  <m-1|S-|m> = sqrt(S(S+1)-m(m-1))
        m_sup = m[:-1]  # m values for column index of S-: m = S, ..., -S+1
        vals_m = torch.sqrt((S * (S + 1) - m_sup * (m_sup - 1)).clamp(min=0.0))
        Sm = torch.diag(vals_m, -1)
    else:
        Sp = torch.zeros(1, 1, dtype=fdtype, device=device)
        Sm = torch.zeros(1, 1, dtype=fdtype, device=device)

    Sz = torch.diag(m)

    if comp == 4:
        return Sp.to(dtype)
    if comp == 5:
        return Sm.to(dtype)
    if comp == 3:
        return Sz.to(dtype)
    if comp == 1:  # Sx = (S+ + S-) / 2
        return (0.5 * (Sp + Sm)).to(dtype)
    if comp == 2:  # Sy = (S+ - S-) / (2i)
        half_j = torch.tensor(-0.5j, dtype=dtype)
        return half_j * (Sp - Sm).to(dtype)

    raise ValueError(f"Unknown component index {comp}")


def sop(
    spins,
    comp,
    dtype: torch.dtype = torch.complex128,
    device: str = 'cpu',
):
    """Build a spin operator matrix for a (multi-)spin system.

    Parameters
    ----------
    spins:
        Sequence of spin quantum numbers, e.g. ``[0.5]``, ``[0.5, 1.0]``.
        A bare float is also accepted for single-spin systems.
    comp:
        Operator specification (multiple formats):

        *String syntax* — one character per spin:
        characters ``e x y z + -`` give the component for each spin in order.
        For a single-spin system the string may be just one character.

        *Integer syntax* — a single int 0–5 (0=e 1=x 2=y 3=z 4=+ 5=-) for
        single-spin systems (mirrors EasySpin's numeric shorthand).

        *Flat-pair syntax* — a 2-element list ``[spin_index, comp_index]``
        (1-based spin index) for selecting one spin's component in a multi-spin
        system.  All other spins receive the identity operator.

        *Multi-pair syntax* — a list of ``[spin_index, comp_index]`` pairs for
        product operators, e.g. ``[[1, 1], [2, 3]]``.

        *List-of-strings syntax* — a list of single-character strings
        (e.g. ``['x', 'y', 'z']``) to return a *list* of operator matrices.

    dtype:
        Returned dtype.  Default ``torch.complex128``.
    device:
        PyTorch device string.  Default ``'cpu'``.

    Returns
    -------
    torch.Tensor or list[torch.Tensor]
        Square matrix of shape ``(nStates, nStates)``, or a list of matrices
        when *comp* is a list of strings.

    Examples
    --------
    >>> Sx = sop(0.5, 'x')               # 2×2 Sx (single spin)
    >>> Sx = sop(0.5, 1)                 # same, numeric index
    >>> Sx, Sy, Sz = sop(0.5, ['x', 'y', 'z'])   # list syntax
    >>> SxI = sop([0.5, 1.0], 'xe')     # 6×6: Sx⊗I
    >>> SxI = sop([0.5, 0.5], [1, 1])   # same, flat-pair [spin_idx=1, comp_idx=x]
    >>> SzIz = sop([0.5, 0.5], [[1, 3], [2, 3]])  # product Sz1⊗Sz2
    """
    # Normalise spins to list of floats
    if isinstance(spins, (int, float)):
        spins = [float(spins)]
    else:
        spins = [float(s) for s in spins]

    # Validate spin quantum numbers
    for s in spins:
        if s < 0:
            raise ValueError(
                f"Invalid spin quantum number {s}: must be non-negative (0, 1/2, 1, ...)."
            )

    n_spins = len(spins)

    # List-of-strings → return a list of operator matrices
    if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], str):
        return [sop(spins, c, dtype=dtype, device=device) for c in comp]

    # Parse component specification -> components[i] = component index for spin i
    components = [0] * n_spins  # default: identity everywhere

    if isinstance(comp, int):
        # Single integer shorthand (0=e, 1=x, 2=y, 3=z, 4=+, 5=-)
        if n_spins != 1:
            raise ValueError(
                f"Single-integer comp={comp} is only valid for single-spin systems "
                f"(n_spins={n_spins}).  Use list format for multi-spin."
            )
        components[0] = comp

    elif isinstance(comp, str):
        comp_str = comp.lower()
        if len(comp_str) == 1:
            if len(spins) == 1:
                pass  # single-spin shorthand
            else:
                raise ValueError(
                    f"comp='{comp}' is a single character but spins has {n_spins} entries. "
                    "Use a string with one character per spin (e.g. 'xe')."
                )
        if len(comp_str) != n_spins:
            raise ValueError(
                f"comp string '{comp}' has {len(comp_str)} characters but spins has {n_spins} entries."
            )
        for i, ch in enumerate(comp_str):
            if ch not in _CHAR_TO_IDX:
                raise ValueError(f"Unknown component character '{ch}'. Use e x y z + -")
            components[i] = _CHAR_TO_IDX[ch]

    else:
        # List-based numeric spec
        comp = list(comp)
        if len(comp) == 0:
            pass  # all identity
        elif isinstance(comp[0], (list, tuple)):
            # Multi-pair syntax: [[spin_idx, comp_idx], ...]
            for pair in comp:
                idx, cidx = int(pair[0]), int(pair[1])
                if idx < 1 or idx > n_spins:
                    raise ValueError(
                        f"Spin index {idx} is out of range for a {n_spins}-spin system (1-based)."
                    )
                components[idx - 1] = cidx
        elif isinstance(comp[0], int):
            # Flat-pair syntax: [spin_idx, comp_idx]
            if len(comp) != 2:
                raise ValueError(
                    f"Flat-pair comp must have exactly 2 elements [spin_idx, comp_idx], "
                    f"got {len(comp)}.  For multiple pairs use [[spin_idx, comp_idx], ...]."
                )
            idx, cidx = int(comp[0]), int(comp[1])
            if idx < 1 or idx > n_spins:
                raise ValueError(
                    f"Spin index {idx} is out of range for a {n_spins}-spin system (1-based)."
                )
            components[idx - 1] = cidx
        else:
            raise ValueError(f"Unrecognised comp format: {comp!r}")

    # Build the full operator via iterated Kronecker products
    Op = torch.tensor([[1.0 + 0j]], dtype=dtype, device=device)
    for i, (S, cidx) in enumerate(zip(spins, components)):
        M = _single_sop(S, cidx, dtype=dtype, device=device)
        Op = torch.kron(Op, M)

    return Op


def commute(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return the commutator [A, B] = A @ B - B @ A.

    Parameters
    ----------
    A, B:
        Square matrices of the same shape (torch.Tensor).

    Returns
    -------
    C : torch.Tensor
        Commutator matrix.

    Examples
    --------
    >>> import torch
    >>> Sx, Sz = sop(0.5, 'x'), sop(0.5, 'z')
    >>> commute(Sx, Sz)   # should equal -1j * Sy  (within sign convention)
    """
    if A.shape != B.shape:
        raise ValueError(f"commute: A and B must have the same shape, got {A.shape} vs {B.shape}")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("commute: A and B must be square matrices")
    return A @ B - B @ A
