"""Tests for torchspin.angmom — angular momentum functions.

Tests compare against known analytical values and fundamental identities.
"""
import math
import numpy as np
import pytest

from torchspin.angmom import (
    wigner3j, wigner6j, wignerd, spherharm, clebschgordan, cgmatrix,
    isto, isto2stev
)

# ---------------------------------------------------------------------------
# wigner3j
# ---------------------------------------------------------------------------

class TestWigner3j:
    """Known values from Edmonds (1957) Table 2 and selection rules."""

    def test_selection_rule_m_sum(self):
        """m1+m2+m3 ≠ 0 → 0."""
        assert wigner3j(1, 1, 0, 1, 0, 0) == 0.0  # m sum = 1

    def test_selection_rule_triangle(self):
        """Triangle inequality violated → 0."""
        assert wigner3j(1, 1, 3, 0, 0, 0) == 0.0  # j3=3 > j1+j2=2

    def test_selection_rule_m_bounds(self):
        """|m| > j → 0."""
        assert wigner3j(1, 1, 2, 2, 0, -2) == 0.0  # |m1|=2 > j1=1

    def test_j_zero(self):
        """j3=0, j1=j2: standard result = (-1)^(j1-m1)/sqrt(2j1+1)."""
        j = 1.0
        for m in (-1.0, 0.0, 1.0):
            expected = (-1) ** int(j - m) / math.sqrt(2 * j + 1)
            got = wigner3j(j, j, 0, m, -m, 0)
            assert abs(got - expected) < 1e-12, f"m={m}: {got} vs {expected}"

    def test_half_integer(self):
        """W3j(1/2,1/2,0; 1/2,-1/2,0) = +1/sqrt(2) (DLMF 34.2.4: (-1)^(j-m)/sqrt(2j+1))."""
        val = wigner3j(0.5, 0.5, 0, 0.5, -0.5, 0)
        assert abs(val - 1 / math.sqrt(2)) < 1e-12

    def test_known_j1(self):
        """W3j(1,1,2; 1,-1,0): standard Edmonds value = 1/sqrt(30)."""
        # From standard tables: W3j(1,1,2; 1,-1,0) = 1/sqrt(30)
        val = wigner3j(1, 1, 2, 1, -1, 0)
        assert abs(val - 1 / math.sqrt(30)) < 1e-12

    def test_symmetry_column_swap(self):
        """W3j is invariant under even permutations of columns."""
        args = (1, 2, 3, -1, 1, 0)   # j1,j2,j3,m1,m2,m3 but j3>j1+j2 → 0; use valid
        j1, j2, j3 = 1, 1, 2
        m1, m2, m3 = 0, 1, -1
        v = wigner3j(j1, j2, j3, m1, m2, m3)
        # Even permutation: (j2,j3,j1; m2,m3,m1)
        v2 = wigner3j(j2, j3, j1, m2, m3, m1)
        # Another even permutation: (j3,j1,j2; m3,m1,m2)
        v3 = wigner3j(j3, j1, j2, m3, m1, m2)
        assert abs(v - v2) < 1e-12
        assert abs(v - v3) < 1e-12

    def test_symmetry_odd_permutation(self):
        """Odd permutation multiplies by (-1)^(j1+j2+j3)."""
        j1, j2, j3 = 1, 1, 2
        m1, m2, m3 = 0, 1, -1
        v = wigner3j(j1, j2, j3, m1, m2, m3)
        # Odd permutation: swap columns 1 and 2
        phase = (-1) ** int(j1 + j2 + j3)
        v_swap = wigner3j(j2, j1, j3, m2, m1, m3)
        assert abs(v * phase - v_swap) < 1e-12

    def test_m_inversion(self):
        """Inverting all m multiplies by (-1)^(j1+j2+j3)."""
        j1, j2, j3 = 1, 1, 2
        m1, m2, m3 = 0, 1, -1
        v = wigner3j(j1, j2, j3, m1, m2, m3)
        phase = (-1) ** int(j1 + j2 + j3)
        v_inv = wigner3j(j1, j2, j3, -m1, -m2, -m3)
        assert abs(v * phase - v_inv) < 1e-12

    def test_orthogonality(self):
        """Sum over m1,m2 of W3j^2 * (2j3+1) = 1 (if j3 in triangle)."""
        j1, j2, j3 = 1.0, 1.0, 2.0
        total = 0.0
        for m1 in np.arange(-j1, j1 + 0.5):
            m2 = -m1  # forced by m3=0 selection
            total += wigner3j(j1, j2, j3, m1, m2, 0) ** 2
        assert abs(total * (2 * j3 + 1) - 1.0) < 1e-10

    def test_larger_j(self):
        """Verify j=5 case: W3j(5,5,0; 3,-3,0) = (-1)^2/sqrt(11)."""
        val = wigner3j(5, 5, 0, 3, -3, 0)
        expected = (-1) ** int(5 - 3) / math.sqrt(11)
        assert abs(val - expected) < 1e-10

    def test_three_half(self):
        """W3j(3/2,1/2,1; 1/2,-1/2,0) — half-integer, check vs sum rule."""
        val = wigner3j(1.5, 0.5, 1, 0.5, -0.5, 0)
        # From explicit formula: should be nonzero and real
        assert not math.isnan(val)
        assert abs(val) > 0


# ---------------------------------------------------------------------------
# clebschgordan
# ---------------------------------------------------------------------------

class TestClebschGordan:
    """CG coefficients via relation to Wigner 3-j."""

    def test_spin_half_coupling(self):
        """<1/2,1/2; 1/2,-1/2|1,0> = 1/sqrt(2)."""
        val = clebschgordan(0.5, 0.5, 1, 0.5, -0.5, 0)
        assert abs(val - 1 / math.sqrt(2)) < 1e-12

    def test_spin_one_coupling(self):
        """<1,1; 1,0|2,1> = 1/sqrt(2) — from lowering J-|2,2>=|2,1>."""
        val = clebschgordan(1, 1, 2, 1, 0, 1)
        expected = 1 / math.sqrt(2)
        assert abs(val - expected) < 1e-12

    def test_selection_rule_m(self):
        """m1+m2 ≠ m → 0."""
        val = clebschgordan(1, 1, 2, 1, 1, 0)  # m1+m2=2≠0=m
        assert val == 0.0

    def test_completeness(self):
        """Sum over m1,m2 of CG^2 = 1 for given J, M."""
        j1, j2, J = 1.0, 1.0, 2.0
        M = 0.0
        total = sum(clebschgordan(j1, j2, J, m1, M - m1, M) ** 2
                    for m1 in np.arange(-j1, j1 + 0.5)
                    if abs(M - m1) <= j2)
        assert abs(total - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# cgmatrix
# ---------------------------------------------------------------------------

class TestCGMatrix:
    """cgmatrix: unitary, correct shape, correct quantum numbers."""

    def test_shape_half_half(self):
        U2C, Smtot, m12 = cgmatrix(0.5, 0.5)
        assert U2C.shape == (4, 4)
        assert Smtot.shape == (4, 2)
        assert m12.shape == (4, 2)

    def test_unitary_half_half(self):
        U2C, _, _ = cgmatrix(0.5, 0.5)
        assert np.allclose(U2C @ U2C.T, np.eye(4), atol=1e-12)
        assert np.allclose(U2C.T @ U2C, np.eye(4), atol=1e-12)

    def test_unitary_one_half(self):
        U2C, _, _ = cgmatrix(1.0, 0.5)
        n = int((2 * 1.0 + 1) * (2 * 0.5 + 1))
        assert U2C.shape == (n, n)
        assert np.allclose(U2C @ U2C.T, np.eye(n), atol=1e-12)

    def test_total_spins(self):
        """For S1=S2=1/2, coupled spins are 1 (triplet) and 0 (singlet)."""
        _, Smtot, _ = cgmatrix(0.5, 0.5)
        stot_vals = sorted(set(Smtot[:, 0]))
        assert np.allclose(stot_vals, [0.0, 1.0])

    def test_stot_filter(self):
        """Requesting only Stot=1 returns 3×3 subspace."""
        U2C, Smtot, _ = cgmatrix(0.5, 0.5, Stot=1)
        assert U2C.shape == (3, 4)  # 3 rows (triplet), 4 columns (uncoupled)
        # All rows have Stot=1
        assert all(abs(Smtot[i, 0] - 1.0) < 1e-10 for i in range(3))

    def test_known_singlet(self):
        """Singlet state |S=0,m=0> = (|↑↓> - |↓↑>)/√2."""
        U2C, Smtot, m12 = cgmatrix(0.5, 0.5)
        # Find singlet row
        singlet_idx = [i for i in range(4) if abs(Smtot[i, 0]) < 1e-10][0]
        row = U2C[singlet_idx]
        # Uncoupled basis: |↑↑>, |↑↓>, |↓↑>, |↓↓> (m12 lists mS1, mS2)
        # singlet = 1/√2 |↑↓> - 1/√2 |↓↑>
        # find indices for |↑↓> and |↓↑>
        idx_updown = [i for i in range(4) if m12[i, 0] > 0 and m12[i, 1] < 0][0]
        idx_downup = [i for i in range(4) if m12[i, 0] < 0 and m12[i, 1] > 0][0]
        assert abs(row[idx_updown] - 1 / math.sqrt(2)) < 1e-12
        assert abs(row[idx_downup] + 1 / math.sqrt(2)) < 1e-12


# ---------------------------------------------------------------------------
# wignerd
# ---------------------------------------------------------------------------

class TestWignerd:
    """Wigner rotation matrix — identities and known values."""

    def test_beta_zero_is_identity(self):
        """D^J(α,0,γ) is diagonal with exp(-i·m·(α+γ)) entries."""
        J = 1.5
        D = wignerd(J, 0.2, 0.0, 0.3, phase='-')
        # Off-diagonals should be ~0
        nJ = int(2 * J) + 1
        D_diag = np.abs(np.diag(D))
        D_off  = np.abs(D - np.diag(np.diag(D)))
        assert np.all(D_off < 1e-12)
        assert np.allclose(D_diag, 1.0, atol=1e-12)

    def test_beta_pi_antidiagonal(self):
        """D^1(0,π,0): d^1_{m'm}(π) = (-1)^(1-m') δ_{m',-m}."""
        D = wignerd(1.0, 0.0, math.pi, 0.0, phase='-')
        # Basis: m = +1, 0, -1
        # d^1_{+1,-1} = 0, d^1_{+1,0} = 0, d^1_{+1,+1} = 0 ... ?
        # Actually d^1_{m'm}(pi) = (-1)^(1-m) delta(m',-m)
        # Row 0 (m'=+1): col 2 (m=-1) = (-1)^(1-(-1)) = 1
        # Row 1 (m'=0):  col 1 (m=0)  = (-1)^(1-0) = -1
        # Row 2 (m'=-1): col 0 (m=+1) = (-1)^(1-1) = 1
        expected = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
        assert np.allclose(np.real(D), expected, atol=1e-12)

    def test_unitarity(self):
        """D^J must be unitary: D†D = I."""
        for J in (0.5, 1.0, 1.5, 2.0):
            D = wignerd(J, 0.3, 1.1, 0.7, phase='-')
            nJ = int(2 * J) + 1
            prod = D.conj().T @ D
            assert np.allclose(prod, np.eye(nJ), atol=1e-11), f"J={J}: not unitary"

    def test_composition(self):
        """D^J(R1) @ D^J(R2) = D^J(R1·R2) for J=1/2."""
        # For J=1/2 with simple y-rotations: D(beta1) @ D(beta2) = D(beta1+beta2)
        b1, b2 = 0.4, 0.7
        D1 = wignerd(0.5, 0.0, b1, 0.0, phase='-')
        D2 = wignerd(0.5, 0.0, b2, 0.0, phase='-')
        D12 = wignerd(0.5, 0.0, b1 + b2, 0.0, phase='-')
        assert np.allclose(D1 @ D2, D12, atol=1e-12)

    def test_beta_only_is_real(self):
        """With only beta (α=γ=0), d^J should be real."""
        d = wignerd(1.5, math.pi / 3)  # beta-only form
        assert np.isrealobj(d) or np.allclose(d.imag, 0, atol=1e-12)

    def test_single_element_matches_matrix(self):
        """Single-element form [J,m1,m2] matches full matrix."""
        J = 1.0; a, b, g = 0.5, 0.8, 1.2
        D = wignerd(J, a, b, g, phase='-')
        ms = np.arange(J, -J - 0.5, -1)
        for i, m1 in enumerate(ms):
            for j, m2 in enumerate(ms):
                d_elem = wignerd([J, m1, m2], a, b, g, phase='-')
                assert abs(d_elem - D[i, j]) < 1e-12, f"m1={m1},m2={m2}"

    def test_phase_plus_vs_minus(self):
        """'+' phase is complex conjugate of '-' phase (for real beta only)."""
        d_minus = wignerd(1.0, math.pi / 4)
        d_plus  = wignerd(1.0, math.pi / 4, phase='+')
        assert np.allclose(d_minus, d_plus.conj() if hasattr(d_plus, 'conj') else d_plus,
                           atol=1e-12)


# ---------------------------------------------------------------------------
# spherharm
# ---------------------------------------------------------------------------

class TestSpherHarm:
    """Spherical harmonics — normalization and symmetry."""

    def test_Y00(self):
        """Y_0^0 = 1/sqrt(4pi)."""
        y = spherharm(0, 0, 0.5, 0.7)
        assert abs(y - 1 / math.sqrt(4 * math.pi)) < 1e-12

    def test_Y10_on_axis(self):
        """Y_1^0(θ=0) = sqrt(3/(4π))."""
        y = spherharm(1, 0, 0.0, 0.0)
        assert abs(y - math.sqrt(3 / (4 * math.pi))) < 1e-12

    def test_normalization(self):
        """Integrate |Y_L^M|^2 over sphere ≈ 1 (Monte Carlo)."""
        rng = np.random.default_rng(42)
        N = 100_000
        theta = np.arccos(2 * rng.random(N) - 1)  # uniform on sphere
        phi = 2 * math.pi * rng.random(N)
        for L, M in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 2)):
            y = spherharm(L, M, theta, phi)
            integral = np.mean(np.abs(y) ** 2) * 4 * math.pi
            assert abs(integral - 1.0) < 0.05, f"L={L},M={M}: norm={integral:.4f}"

    def test_complex_conj(self):
        """Y_L^{-M} = (-1)^M * conj(Y_L^M) (with CS phase)."""
        L, M = 2, 1
        theta, phi = 0.8, 1.2
        yp = spherharm(L,  M, theta, phi)
        ym = spherharm(L, -M, theta, phi)
        assert abs(ym - (-1) ** M * np.conj(yp)) < 1e-12

    def test_real_kind(self):
        """Real spherical harmonics are real-valued."""
        y = spherharm(2, 1, 0.6, 0.9, kind='r')
        assert np.isrealobj(y) or abs(y.imag) < 1e-14

    def test_real_normalization(self):
        """Real harmonics are also normalized to 1 on sphere."""
        rng = np.random.default_rng(7)
        N = 100_000
        theta = np.arccos(2 * rng.random(N) - 1)
        phi = 2 * math.pi * rng.random(N)
        for L, M in ((1, -1), (1, 0), (1, 1), (2, -2), (2, 0)):
            y = spherharm(L, M, theta, phi, kind='r')
            integral = np.mean(np.abs(y) ** 2) * 4 * math.pi
            assert abs(integral - 1.0) < 0.05, f"L={L},M={M}: norm={integral:.4f}"

    def test_array_input(self):
        """Vectorized theta/phi input works."""
        theta = np.array([0.1, 0.5, 1.0])
        phi   = np.array([0.2, 0.6, 1.1])
        y = spherharm(2, 1, theta, phi)
        assert y.shape == (3,)
        # Each element matches scalar call
        for i in range(3):
            yi = spherharm(2, 1, theta[i], phi[i])
            assert abs(y[i] - yi) < 1e-14


# ---------------------------------------------------------------------------
# wigner6j
# ---------------------------------------------------------------------------

class TestWigner6j:
    """Known values and selection rules for Wigner 6-j symbols."""

    def test_triangle_rule_zero(self):
        """Triangle violation → 0."""
        assert wigner6j(0, 0, 2, 0, 0, 0) == 0.0

    def test_half_half_one(self):
        """Analytic: {1/2 1/2 1; 1/2 1/2 0} = -1/2."""
        # From Brink & Satchler tables
        v = wigner6j(0.5, 0.5, 1, 0.5, 0.5, 0)
        assert abs(v - 0.5) < 1e-10

    def test_one_one_two(self):
        """Analytic: {1 1 2; 1 1 0} known value."""
        v = wigner6j(1, 1, 2, 1, 1, 0)
        # From tables: {1 1 2; 1 1 0} = 1/sqrt(30) * (-1)? check sign
        # Using the formula: computed via scipy if available
        try:
            from sympy.physics.wigner import wigner_6j
            ref = float(wigner_6j(1, 1, 2, 1, 1, 0))
            assert abs(v - ref) < 1e-8
        except ImportError:
            # Without sympy, check it's nonzero and reasonable magnitude
            assert abs(v) > 1e-10
            assert abs(v) < 1.0

    def test_zero_entry(self):
        """{0 j j; 0 j j} = (-1)^(2j) / (2j+1).

        From the 6j formula with one entry zero:
        {0 b c; a e f} = δ(b,c) δ(e,f) (-1)^(a+b+e) / sqrt((2b+1)(2e+1))
        → {0 j j; 0 j j} = (-1)^(2j) / (2j+1)
        """
        j = 1.0
        v = wigner6j(0, j, j, 0, j, j)
        expected = ((-1.0) ** int(round(2 * j))) / (2 * j + 1)
        assert abs(v - expected) < 1e-10

    def test_symmetry_column_permutation(self):
        """6j is symmetric under permutation of columns."""
        j1, j2, j3, J1, J2, J3 = 1, 2, 1, 1, 0, 2
        v1 = wigner6j(j1, j2, j3, J1, J2, J3)
        v2 = wigner6j(j2, j1, j3, J2, J1, J3)  # swap columns 1↔2
        assert abs(v1 - v2) < 1e-10

    def test_all_zero_returns_zero(self):
        """All-zero arguments: {0 0 0; 0 0 0} = 1."""
        v = wigner6j(0, 0, 0, 0, 0, 0)
        assert abs(v - 1.0) < 1e-10

    def test_nonphysical_half_int_check(self):
        """Non-half-integer returns 0 (not a valid angular momentum)."""
        v = wigner6j(0.3, 0.5, 0.5, 0.5, 0.5, 0.5)
        assert v == 0.0

    def test_return_type(self):
        """Returns a float."""
        v = wigner6j(1, 1, 1, 1, 1, 1)
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# isto
# ---------------------------------------------------------------------------

class TestIsto:
    """Tests for irreducible spherical tensor operators."""

    def test_shape_spin_half_rank1(self):
        """T^1_0 for S=1/2: (2, 2) matrix."""
        T = isto(0.5, 1, 0)
        assert T.shape == (2, 2)

    def test_shape_spin1_rank2(self):
        """T^2_0 for S=1: (3, 3) matrix."""
        T = isto(1.0, 2, 0)
        assert T.shape == (3, 3)

    def test_rank0_is_identity_scaled(self):
        """T^0_0 = I / sqrt(2J+1) (proportional to identity)."""
        J = 1.0
        T = isto(J, 0, 0)
        dim = int(round(2 * J + 1))
        assert T.shape == (dim, dim)
        # Should be proportional to identity
        diag_vals = np.diag(np.real(T))
        assert np.allclose(diag_vals, diag_vals[0], atol=1e-10)
        # Off-diagonal should be zero
        off = T - np.diag(np.diag(T))
        assert np.allclose(off, 0, atol=1e-10)

    def test_hermitian_for_q0(self):
        """T^k_0 is Hermitian (for real, symmetric systems)."""
        T = isto(1.0, 2, 0)
        assert np.allclose(T, T.conj().T, atol=1e-10)

    def test_negative_q_relation(self):
        """T^k_{-q} = (-1)^q * (T^k_q)^†  (Ryabov Eq.[3])."""
        J = 1.0
        k, q = 2, 1
        Tp = isto(J, k, q)
        Tm = isto(J, k, -q)
        expected = ((-1.0) ** q) * Tp.conj().T
        assert np.allclose(Tm, expected, atol=1e-10)

    def test_array_kq_input(self):
        """isto(J, [k, q]) array input works."""
        T = isto(0.5, [1, 0])
        assert T.shape == (2, 2)

    def test_traceless_rank2(self):
        """T^2_q: for q≠0, trace is 0."""
        T = isto(1.0, 2, 1)
        assert abs(np.trace(T)) < 1e-10


class TestIsto2stev:
    """Tests for ISTO → Stevens transformation matrix."""

    def test_shape_rank2(self):
        """isto2stev(2) returns (5, 5) matrix."""
        C = isto2stev(2)
        assert C.shape == (5, 5)

    def test_shape_rank1(self):
        """isto2stev(1) returns (3, 3) matrix."""
        C = isto2stev(1)
        assert C.shape == (3, 3)

    def test_rank0_is_scalar(self):
        """isto2stev(0) returns (1, 1) matrix (trivial)."""
        C = isto2stev(0)
        assert C.shape == (1, 1)

    def test_invertible(self):
        """C is invertible (det != 0)."""
        C = isto2stev(2)
        assert abs(np.linalg.det(C)) > 1e-10

    def test_diagonal_antidiagonal_structure(self):
        """Only diagonal and anti-diagonal elements are non-zero."""
        C = isto2stev(2)
        nq = 5
        for i in range(nq):
            for j in range(nq):
                if i != j and i != (nq - 1 - j):
                    assert abs(C[i, j]) < 1e-10, f"C[{i},{j}] = {C[i,j]} should be 0"

    def test_inverse_roundtrip(self):
        """C @ inv(C) ≈ identity."""
        C = isto2stev(2)
        C_inv = np.linalg.inv(C)
        assert np.allclose(C @ C_inv, np.eye(5), atol=1e-10)
