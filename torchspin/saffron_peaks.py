"""saffron_peaks — Peak accumulation and time-domain evolution for pulse EPR.

Port of EasySpin's ``sf_peaks.c`` (frequency-domain DFT binning) and
``sf_evolve`` (time-domain direct accumulation).

All functions operate on complex128 PyTorch tensors.
"""

import torch
import math

from torchspin._compile import maybe_compile


# ---------------------------------------------------------------------------
# DFT bin index computation (matches sf_peaks.c exactly)
# ---------------------------------------------------------------------------
@maybe_compile
def _dft_bin(nu: torch.Tensor, dt: float, n_points: int) -> torch.Tensor:
    """Map frequency to DFT bin index with correct quantum-phase sign.

    The minus sign accounts for the convention that quantum evolution
    uses exp(-i*omega*t) while DFT uses exp(+i*omega*t).

    Parameters
    ----------
    nu : Tensor
        Nuclear transition frequencies (MHz).
    dt : float
        Time step (µs).
    n_points : int
        Number of DFT points.

    Returns
    -------
    Tensor of int64
        Bin indices in [0, n_points).
    """
    frac = torch.fmod(-nu * dt, 1.0) * n_points
    # MATLAB convention: floor for positive, ceil for negative, wrap negatives
    idx = torch.where(frac >= 0, frac.floor(), frac.ceil()).long()
    idx = idx % n_points  # wrap negative indices
    return idx


# ---------------------------------------------------------------------------
# Frequency-domain peak accumulation (replaces sf_peaks.c)
# ---------------------------------------------------------------------------
def sf_peaks(inc_scheme_id: int,
             buff_re: torch.Tensor,
             buff_im: torch.Tensor,
             dt: list[float],
             free_l: list[int],
             free_r: list[int],
             Ea: torch.Tensor,
             Eb: torch.Tensor,
             G: torch.Tensor,
             D: torch.Tensor,
             *mixing_mats: torch.Tensor) -> None:
    """Accumulate peaks into frequency-domain buffer (in-place).

    This is a direct port of EasySpin's sf_peaks.c mex function.

    Parameters
    ----------
    inc_scheme_id : int
        Incrementation scheme identifier (1, 2, 3, 11, ...).
    buff_re, buff_im : Tensor
        Real and imaginary spectral buffers to accumulate into (modified in-place).
        Shape: ``(nPoints1,)`` for 1D or ``(nPoints1, nPoints2)`` for 2D.
    dt : list of float
        Time step(s) in µs, one per dimension.
    free_l, free_r : list of int
        Left/right manifold indices for each free evolution (1=alpha, 2=beta).
    Ea, Eb : Tensor
        Eigenvalues for alpha and beta manifolds, shape ``(nStates,)``.
    G : Tensor
        Start coherence matrix, shape ``(nStates, nStates)``.
    D : Tensor
        Detection matrix, shape ``(nStates, nStates)``.
    *mixing_mats : Tensor
        Mixing matrices T1l, T1r [, T2l, T2r, ...] depending on scheme.
    """
    E = [Ea, Eb]  # 1-indexed via free_l/free_r values (1 or 2)

    if inc_scheme_id == 1:
        # [1]: e.g., 3-pulse ESEEM — 1 free evolution
        n_points = buff_re.shape[0]
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]

        # Frequencies: nu[j,k] = E1l[j] - E1r[k]
        nu = E1l.unsqueeze(1) - E1r.unsqueeze(0)  # (N, N)
        idx = _dft_bin(nu, dt[0], n_points)

        # Amplitudes: Amp[j,k] = D[k,j] * G[j,k]  (note MATLAB column-major: D[kj] * G[jk])
        amp = D.T * G  # element-wise product of transposed D with G

        buff_re.scatter_add_(0, idx.reshape(-1), amp.real.reshape(-1))
        buff_im.scatter_add_(0, idx.reshape(-1), amp.imag.reshape(-1))

    elif inc_scheme_id == 2:
        # [1 1]: e.g., 2-pulse ESEEM — 2 free evolutions, 1D
        n_points = buff_re.shape[0]
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]
        E2l = E[free_l[1] - 1]
        E2r = E[free_r[1] - 1]
        N = len(Ea)

        # Amp[i,j,k,l] = T1l[i,j] * G[j,k] * T1r[k,l] * D[l,i]
        # nu = (E2l[i] - E2r[l]) + (E1l[j] - E1r[k])
        # Use einsum for the rank-4 contraction
        # Note: MATLAB uses column-major indexing, so T1l[ij] = T1l(i,j) = T1l[i + j*N]
        # In Python (row-major), T1l[i,j] is already correct.

        # Build 4D amplitude and frequency tensors
        # i,j,k,l all range over N
        amp_4d = (T1l.unsqueeze(2).unsqueeze(3) *       # (N,N,1,1)
                  G.unsqueeze(0).unsqueeze(3) *           # (1,N,N,1)
                  T1r.unsqueeze(0).unsqueeze(1) *         # (1,1,N,N)
                  D.T.unsqueeze(1).unsqueeze(2))          # (N,1,1,N) — D[l,i] = D.T[i,l]
        # Wait — D[l,i] indexing. In the C code: D[li] where li = l + i*N (column-major)
        # So D[l,i] in MATLAB column-major = D(l+1, i+1) = our D[l, i]
        # The 4D tensor is indexed [i,j,k,l], and D factor is D[l,i]

        # Let me redo this more carefully:
        # For indices (i,j,k,l):
        #   amp = T1l[i,j] * G[j,k] * T1r[k,l] * D[l,i]
        #   nu  = (E2l[i] - E2r[l]) + (E1l[j] - E1r[k])

        # Compute amplitude via einsum
        # T1l[i,j] * G[j,k] → (i,k) contraction over j, but we need all 4 indices
        # Better: compute as outer products and element-wise multiply
        amp_ijkl = torch.einsum('ij,jk,kl,li->ijkl', T1l, G, T1r, D)
        nu_ijkl = ((E2l.reshape(-1,1,1,1) - E2r.reshape(1,1,1,-1)) +
                    (E1l.reshape(1,-1,1,1) - E1r.reshape(1,1,-1,1)))
        idx = _dft_bin(nu_ijkl, dt[0], n_points)

        buff_re.scatter_add_(0, idx.reshape(-1), amp_ijkl.real.reshape(-1))
        buff_im.scatter_add_(0, idx.reshape(-1), amp_ijkl.imag.reshape(-1))

    elif inc_scheme_id == 3:
        # [1 -1]: custom, 2 free evolutions, 1D
        n_points = buff_re.shape[0]
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]
        E2l = E[free_l[1] - 1]
        E2r = E[free_r[1] - 1]

        amp_ijkl = torch.einsum('ij,jk,kl,li->ijkl', T1l, G, T1r, D)
        # From sf_peaks.c case 3: nu = -E2l[i] + E1l[j] - E1r[k] + E2r[l]
        nu_ijkl = (-E2l.reshape(-1,1,1,1) + E1l.reshape(1,-1,1,1)
                   - E1r.reshape(1,1,-1,1) + E2r.reshape(1,1,1,-1))
        idx = _dft_bin(nu_ijkl, dt[0], n_points)

        buff_re.scatter_add_(0, idx.reshape(-1), amp_ijkl.real.reshape(-1))
        buff_im.scatter_add_(0, idx.reshape(-1), amp_ijkl.imag.reshape(-1))

    elif inc_scheme_id == 11:
        # [1 2]: HYSCORE — 2 free evolutions, 2D
        n_points1, n_points2 = buff_re.shape
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        E1l = E[free_l[0] - 1]
        E1r = E[free_r[0] - 1]
        E2l = E[free_l[1] - 1]
        E2r = E[free_r[1] - 1]
        N = len(Ea)

        # Pre-compute nuclear frequency bin indices (like sf_peaks.c)
        # nu1[i,j] for dimension 1, nu2[i,j] for dimension 2
        nu1_ij = E1l.unsqueeze(1) - E1r.unsqueeze(0)  # (N, N)
        nu2_ij = E2l.unsqueeze(1) - E2r.unsqueeze(0)  # (N, N)
        idx1 = _dft_bin(nu1_ij, dt[0], n_points1)
        idx2 = _dft_bin(nu2_ij, dt[1], n_points2)

        # Amp[i,j,k,l] = T1l[i,j] * G[j,k] * T1r[k,l] * D[l,i]
        # Binned at: (nu1[j,k], nu2[i,l])
        # In sf_peaks.c: idx = nu1[jk] + nu2[il]*nPoints1
        amp_ijkl = torch.einsum('ij,jk,kl,li->ijkl', T1l, G, T1r, D)

        # Flatten and compute 2D indices: idx1 uses (j,k) and idx2 uses (i,l)
        # Map each (i,j,k,l) to (idx1[j,k], idx2[i,l])
        idx1_4d = idx1.unsqueeze(0).unsqueeze(3).expand(N, N, N, N)  # j,k dims
        idx2_4d = idx2.unsqueeze(1).unsqueeze(2).expand(N, N, N, N)  # i,l dims

        # Flatten for scatter_add_ on flattened 2D buffer
        flat_idx = (idx1_4d * n_points2 + idx2_4d).reshape(-1)
        buff_re.reshape(-1).scatter_add_(0, flat_idx, amp_ijkl.real.reshape(-1))
        buff_im.reshape(-1).scatter_add_(0, flat_idx, amp_ijkl.imag.reshape(-1))

    elif inc_scheme_id == 12:
        # [1 2 1]: 3 free evolutions, 2D (e.g. 2D three-pulse ESEEM)
        # Amp = D[ni]*T2l[ij]*T1l[jk]*G[kl]*T1r[lm]*T2r[mn]
        # nu1 = E1l[k]-E1r[l] + E3l[i]-E3r[n]
        # nu2 = E2l[j]-E2r[m]
        n_points1, n_points2 = buff_re.shape
        T1l, T1r, T2l, T2r = mixing_mats[0], mixing_mats[1], mixing_mats[2], mixing_mats[3]
        N = len(Ea)
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]

        # Full rank-6 contraction: Amp[i,j,k,l,m,n] = D[ni]*T2l[ij]*T1l[jk]*G[kl]*T1r[lm]*T2r[mn]
        amp = torch.einsum('ij,jk,kl,lm,mn,ni->ijklmn', T2l, T1l, G, T1r, T2r, D)

        nu1 = (El[0].reshape(1,1,-1,1,1,1) - Er[0].reshape(1,1,1,-1,1,1) +
               El[2].reshape(-1,1,1,1,1,1) - Er[2].reshape(1,1,1,1,1,-1))
        nu2 = El[1].reshape(1,-1,1,1,1,1) - Er[1].reshape(1,1,1,1,-1,1)

        idx1 = _dft_bin(nu1, dt[0], n_points1)
        idx2 = _dft_bin(nu2, dt[1], n_points2)
        flat_idx = (idx1 * n_points2 + idx2).reshape(-1)
        buff_re.reshape(-1).scatter_add_(0, flat_idx, amp.real.reshape(-1))
        buff_im.reshape(-1).scatter_add_(0, flat_idx, amp.imag.reshape(-1))

    elif inc_scheme_id == 14:
        # [1 1 2]: 3 free evolutions, 2D (e.g. Hubrich CF-NF(2))
        # Amp = D[ni]*T2l[ij]*T1l[jk]*G[kl]*T1r[lm]*T2r[mn]
        # nu1 = E1l[k]-E1r[l] + E2l[j]-E2r[m]
        # nu2 = E3l[i]-E3r[n]
        n_points1, n_points2 = buff_re.shape
        T1l, T1r, T2l, T2r = mixing_mats[0], mixing_mats[1], mixing_mats[2], mixing_mats[3]
        N = len(Ea)
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]

        amp = torch.einsum('ij,jk,kl,lm,mn,ni->ijklmn', T2l, T1l, G, T1r, T2r, D)

        nu1 = (El[0].reshape(1,1,-1,1,1,1) - Er[0].reshape(1,1,1,-1,1,1) +
               El[1].reshape(1,-1,1,1,1,1) - Er[1].reshape(1,1,1,1,-1,1))
        nu2 = El[2].reshape(-1,1,1,1,1,1) - Er[2].reshape(1,1,1,1,1,-1)

        idx1 = _dft_bin(nu1, dt[0], n_points1)
        idx2 = _dft_bin(nu2, dt[1], n_points2)
        flat_idx = (idx1 * n_points2 + idx2).reshape(-1)
        buff_re.reshape(-1).scatter_add_(0, flat_idx, amp.real.reshape(-1))
        buff_im.reshape(-1).scatter_add_(0, flat_idx, amp.imag.reshape(-1))

    elif inc_scheme_id == 15:
        # [1 2 2 1]: 4 free evolutions, 2D (e.g. 2D-CP)
        # Amp = D[pi]*T3l[ij]*T2l[jk]*T1l[kl]*G[lm]*T1r[mn]*T2r[no]*T3r[op]
        # nu1 = E1l[l]-E1r[m] + E4l[i]-E4r[p]
        # nu2 = E2l[k]-E2r[n] + E3l[j]-E3r[o]
        n_points1, n_points2 = buff_re.shape
        T1l, T1r, T2l, T2r, T3l, T3r = (mixing_mats[0], mixing_mats[1],
                                           mixing_mats[2], mixing_mats[3],
                                           mixing_mats[4], mixing_mats[5])
        N = len(Ea)
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]

        # Rank-8 is too large to expand fully. Use nested einsum.
        # Inner: C[kl,mn] = T1l[kl]*G[lm]*T1r[mn]
        C = torch.einsum('kl,lm,mn->klmn', T1l, G, T1r)
        # Middle: B[jk,no] = T2l[jk]*C[klmn]*T2r[no]  contracted over l,m
        # B[j,k,m,n,o] but we need to be careful
        # Actually: Amp = D[pi]*T3l[ij]*T2l[jk]*T1l[kl]*G[lm]*T1r[mn]*T2r[no]*T3r[op]
        # Contract as: A1 = T2l @ T1l @ G @ T1r @ T2r (chain of matrix products)
        # A1[j,n] = sum_{k,l,m} T2l[jk]*T1l[kl]*G[lm]*T1r[mn]  -- wait, T1r index is [mn]
        # Let's just use einsum:
        A_inner = torch.einsum('jk,kl,lm,mn->jn', T2l, T1l, G, T1r)  # (N,N)
        # Now: Amp[i,j,n,o,p] = D[pi]*T3l[ij]*(A_inner @ T2r)[j,o]*T3r[op]
        A2 = A_inner @ T2r  # (N,N): A2[j,o]
        # Amp[i,j,o,p] = D[pi]*T3l[ij]*A2[j,o]*T3r[op]
        amp_4d = torch.einsum('ij,jo,op,pi->ijop', T3l, A2, T3r, D)

        # But we need the full index structure for frequency computation.
        # nu1 needs l,m,i,p indices and nu2 needs k,n,j,o indices.
        # The einsum above contracts over k,l,m,n — losing them.
        # We need the full rank-8 amplitude, which is N^8 elements.
        # For small N (2-4), this is feasible.
        amp_full = torch.einsum('ij,jk,kl,lm,mn,no,op,pi->ijklmnop',
                                T3l, T2l, T1l, G, T1r, T2r, T3r, D)
        # i,j,k,l,m,n,o,p → 8 indices
        # nu1 = E1l[l]-E1r[m] + E4l[i]-E4r[p]
        # nu2 = E2l[k]-E2r[n] + E3l[j]-E3r[o]
        sh = [1]*8
        def _idx(pos, N):
            s = [1]*8; s[pos] = N; return s

        nu1 = (El[0].reshape(_idx(3, N)) - Er[0].reshape(_idx(4, N)) +
               El[3].reshape(_idx(0, N)) - Er[3].reshape(_idx(7, N)))
        nu2 = (El[1].reshape(_idx(2, N)) - Er[1].reshape(_idx(5, N)) +
               El[2].reshape(_idx(1, N)) - Er[2].reshape(_idx(6, N)))

        idx1 = _dft_bin(nu1, dt[0], n_points1)
        idx2 = _dft_bin(nu2, dt[1], n_points2)
        flat_idx = (idx1 * n_points2 + idx2).reshape(-1)
        buff_re.reshape(-1).scatter_add_(0, flat_idx, amp_full.real.reshape(-1))
        buff_im.reshape(-1).scatter_add_(0, flat_idx, amp_full.imag.reshape(-1))

    elif inc_scheme_id == 17:
        # [1 1 2 2]: 4 free evolutions, 2D (e.g. 2D refocused primary ESEEM)
        # Same rank-8 structure as 15 but different frequency assignment
        # nu1 = E1l[l]-E1r[m] + E2l[k]-E2r[n]
        # nu2 = E3l[j]-E3r[o] + E4l[i]-E4r[p]
        n_points1, n_points2 = buff_re.shape
        T1l, T1r, T2l, T2r, T3l, T3r = (mixing_mats[0], mixing_mats[1],
                                           mixing_mats[2], mixing_mats[3],
                                           mixing_mats[4], mixing_mats[5])
        N = len(Ea)
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]

        amp_full = torch.einsum('ij,jk,kl,lm,mn,no,op,pi->ijklmnop',
                                T3l, T2l, T1l, G, T1r, T2r, T3r, D)

        def _idx8(pos, N):
            s = [1]*8; s[pos] = N; return s

        nu1 = (El[0].reshape(_idx8(3, N)) - Er[0].reshape(_idx8(4, N)) +
               El[1].reshape(_idx8(2, N)) - Er[1].reshape(_idx8(5, N)))
        nu2 = (El[2].reshape(_idx8(1, N)) - Er[2].reshape(_idx8(6, N)) +
               El[3].reshape(_idx8(0, N)) - Er[3].reshape(_idx8(7, N)))

        idx1 = _dft_bin(nu1, dt[0], n_points1)
        idx2 = _dft_bin(nu2, dt[1], n_points2)
        flat_idx = (idx1 * n_points2 + idx2).reshape(-1)
        buff_re.reshape(-1).scatter_add_(0, flat_idx, amp_full.real.reshape(-1))
        buff_im.reshape(-1).scatter_add_(0, flat_idx, amp_full.imag.reshape(-1))

    else:
        raise NotImplementedError(
            f"sf_peaks: IncSchemeID={inc_scheme_id} not yet implemented."
        )


# ---------------------------------------------------------------------------
# Time-domain evolution (replaces sf_evolve in saffron.m)
# ---------------------------------------------------------------------------
def sf_evolve(inc_scheme_id: int,
              n_points: list[int],
              dt: list[float],
              free_l: list[int],
              free_r: list[int],
              Ea: torch.Tensor,
              Eb: torch.Tensor,
              G: torch.Tensor,
              D: torch.Tensor,
              *mixing_mats: torch.Tensor) -> torch.Tensor:
    """Direct time-domain signal accumulation.

    This is autograd-compatible (no scatter_add_ in-place ops).

    Parameters
    ----------
    inc_scheme_id : int
        Incrementation scheme identifier.
    n_points : list of int
        Number of points per dimension.
    dt : list of float
        Time step per dimension (µs).
    free_l, free_r : list of int
        Left/right manifold indices (1=alpha, 2=beta).
    Ea, Eb : Tensor
        Eigenvalues for alpha/beta manifolds.
    G, D : Tensor
        Start coherence and detection matrices.
    *mixing_mats : Tensor
        Mixing matrices.

    Returns
    -------
    Tensor
        Complex time-domain signal, shape ``(nPoints,)`` or ``(nPoints1, nPoints2)``.
    """
    E = [Ea, Eb]
    dtype = torch.complex128
    device = Ea.device
    # Every evolution step multiplies elementwise by a fixed phase matrix
    # UU[i, j] = exp(2πi·dt·(sa·Ea[i] + sb·Eb[j])), so step k carries UU**k.  All
    # steps are formed at once from the exponent (more accurate than repeated
    # multiplication, and one batched product per scheme instead of a Python loop
    # over every time point; up to 0.9.38 the loops were a line-by-line port of
    # EasySpin's sf_evolve, 1.5 s for a 128 × 128 HYSCORE).

    def _steps(n, Ea_, Eb_, sa, sb, dt_):
        """(n, Na, Nb): UU**k for k = 0..n-1."""
        k = torch.arange(n, dtype=torch.float64, device=device).view(-1, 1, 1)
        nu = sa * Ea_.unsqueeze(1) + sb * Eb_.unsqueeze(0)
        return torch.exp((2j * math.pi * dt_) * (k * nu.unsqueeze(0)))

    DT = D.T                                    # detector: Σ_ab D[b, a]·fd[a, b] (MATLAB reshape(D.', 1, NN))

    def _detect_2d(F, P2):
        """signal[k1, k2] = Σ_ab DT[a, b]·P2[k2, a, b]·F[k1, a, b]."""
        n1, n2 = F.shape[0], P2.shape[0]
        return (F * DT).reshape(n1, -1) @ P2.reshape(n2, -1).T

    if inc_scheme_id == 1:
        # [1]: 3-pulse ESEEM (sf_evolve case 1)
        nP = n_points[0] if isinstance(n_points, (list, tuple)) else n_points
        P = _steps(nP, E[free_l[0] - 1], E[free_r[0] - 1], -1.0, +1.0, dt[0])
        return P.reshape(nP, -1) @ (DT * G).reshape(-1)

    elif inc_scheme_id in (2, 3):
        # [1 1]: 2-pulse ESEEM (case 2); [1 -1] (case 3): second dimension decremented
        nP = n_points[0] if isinstance(n_points, (list, tuple)) else n_points
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        E1l, E1r = E[free_l[0] - 1], E[free_r[0] - 1]
        E2l, E2r = E[free_l[1] - 1], E[free_r[1] - 1]
        s2 = -1.0 if inc_scheme_id == 2 else +1.0
        L = _steps(nP, E2l, E1l, s2, -1.0, dt[0]) * T1l
        R = _steps(nP, E1r, E2r, +1.0, -s2, dt[0]) * T1r
        fd = L @ G @ R
        return (fd * DT).sum(dim=(1, 2))

    elif inc_scheme_id == 11:
        # [1 2]: HYSCORE, 2D (case 11)
        nP1, nP2 = n_points
        T1l, T1r = mixing_mats[0], mixing_mats[1]
        P1 = _steps(nP1, E[free_l[0] - 1], E[free_r[0] - 1], -1.0, +1.0, dt[0])
        P2 = _steps(nP2, E[free_l[1] - 1], E[free_r[1] - 1], -1.0, +1.0, dt[1])
        return _detect_2d(T1l @ (P1 * G) @ T1r, P2)

    elif inc_scheme_id in (12, 14):
        # [1 2 1] (dim1: E1 + E3, dim2: E2) or [1 1 2] (dim1: E1 + E2, dim2: E3)
        nP1, nP2 = n_points
        T1l, T1r, T2l, T2r = mixing_mats[0], mixing_mats[1], mixing_mats[2], mixing_mats[3]
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]
        a, b = (2, 1) if inc_scheme_id == 12 else (1, 2)      # second dim-1 evolution, dim-2 evolution
        P1l = _steps(nP1, El[a], El[0], -1.0, -1.0, dt[0])
        P1r = _steps(nP1, Er[0], Er[a], +1.0, +1.0, dt[0])
        P2 = _steps(nP2, El[b], Er[b], -1.0, +1.0, dt[1])
        return _detect_2d(T2l @ (P1l * T1l) @ G @ (P1r * T1r) @ T2r, P2)

    elif inc_scheme_id in (15, 17):
        # [1 2 2 1] (dim1: E1 + E4, dim2: E2 + E3) or [1 1 2 2] (dim1: E1 + E2, dim2: E3 + E4).
        # As in the loop this replaces, the dimension-2 evolution acts through
        # UU2 ⊙ (identity), i.e. only the diagonal phases of UU2 enter.
        nP1, nP2 = n_points
        T1l, T1r, T2l, T2r, T3l, T3r = (mixing_mats[0], mixing_mats[1], mixing_mats[2],
                                        mixing_mats[3], mixing_mats[4], mixing_mats[5])
        El = [E[fl - 1] for fl in free_l]
        Er = [E[fr - 1] for fr in free_r]
        if inc_scheme_id == 15:
            i1, i2a, i2b = 3, 2, 1
            r1, r2a, r2b = 3, 1, 2
        else:
            i1, i2a, i2b = 1, 3, 2
            r1, r2a, r2b = 1, 2, 3
        P1l = _steps(nP1, El[i1], El[0], -1.0, -1.0, dt[0])
        P1r = _steps(nP1, Er[0], Er[r1], +1.0, +1.0, dt[0])
        dl = torch.diagonal(_steps(nP2, El[i2a], El[i2b], -1.0, -1.0, dt[1]), dim1=1, dim2=2)   # (nP2, N)
        dr = torch.diagonal(_steps(nP2, Er[r2a], Er[r2b], +1.0, +1.0, dt[1]), dim1=1, dim2=2)
        F = T3l @ (T2l @ (P1l * T1l)) @ G @ ((P1r * T1r) @ T2r) @ T3r                      # (nP1, N, N)
        P2 = dl.unsqueeze(2) * dr.unsqueeze(1)                                             # (nP2, N, N)
        return _detect_2d(F, P2)

    else:
        raise NotImplementedError(
            f"sf_evolve: IncSchemeID={inc_scheme_id} not yet implemented."
        )
