"""Strong-field Hamiltonian blocks of DTANSUG.F (cubic d^n, Griffith Table A.27).

Copied verbatim (code and tables) from telser-tools ``telser_tools.tanabe_sugano.blocks``
(telser-tools commit 89ba7be) so that torchspin has no runtime dependency on telser-tools.
Parity with the original is checked in torchspin/tests/test_lft_tanabe_sugano.py when
telser-tools is installed.  The full-determinant route to the same levels is the d^n engine
(:mod:`torchspin.lft._engines.dn`), used through :mod:`torchspin.lft.ddn`.

Every (spin, irrep) block of a d^n ion in a cubic field, in the strong-field
basis t2^a e^b{...}, exactly as typed into DTANSUG.F v2.81 (J. Telser), whose
elements come from

* D. S. McClure, *Solid State Physics* **1959**, 9, 399 (pp. 412-419), and
* J. S. Griffith, *The Theory of Transition-Metal Ions*, Cambridge 1964,
  Appendix 2, Table A.27 (p. 410).

Units and conventions
---------------------
* All energies in cm-1 (or any unit common to B, C, Dq).
* B, C are Racah parameters. The Racah A term is left out: every energy of
  d^n should be increased by n(n-1)/2 * A (A, 3A, 6A, ... 28A for d2 ... d8).
* One-electron cubic field (octahedral, Dq > 0): e(g) at +6Dq, t2(g) at -4Dq,
  so a configuration t2^a e^b sits at (6b - 4a) Dq.
* d^n with n > 5 is treated as d^(10-n) holes (DTANSUG.F ll. 78-82): the hole
  sees -Dq (``SIGNXF``), a few off-diagonal elements change sign (``SIGNEL``,
  the hole phase convention of Griffith Table A.27), and the constant
  ``DIAG`` (ll. 1420, 1459, 1533) shifts the hole energies onto the
  d^n scale. d5 needs neither.
* Tetrahedral: the cubic field changes sign (l. 86), so Td d^n with
  Dq_Td > 0 equals Oh d^n with Dq = -Dq_Td.  Irreps keep their names
  (g/u subscripts dropped).

The matrix element lambdas below are transcribed mechanically from the
Dq loop of DTANSUG.F (ll. 1408-1883); each carries its Fortran line number.
The Racah-B loop (ll. 798-1263) uses identical elements, which was verified by
textual comparison of both loops (only the Dq sign treatment differs; see
docs/tanabe_sugano.md, finding 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import numpy as np

__all__ = [
    "Block", "BLOCKS", "blocks_for", "get_block", "block_for_nfig", "cubic_sign",
    "electron_hole_sign", "diag_offset", "hamiltonian", "GEOMETRIES",
]

R2, R3, R6 = sqrt(2.0), sqrt(3.0), sqrt(6.0)
GEOMETRIES = ("Oh", "Td")

Element = tuple[int, int, int, Callable[[float, float, float, float], float]]


@dataclass(frozen=True)
class Block:
    """One (2S+1)Gamma block of the d^n (n <= 5 electrons or holes) problem.

    Attributes
    ----------
    nh : number of d electrons (n <= 5) or holes (10 - n) the block refers to.
    first : Fortran index (``NGRP``) of the first strong-field configuration;
        the menu number ``NFIG`` of DTANSUG.F selects the block containing it.
    label : e.g. ``"3T1"`` (multiplicity, irrep).
    terms : free-ion terms the block correlates with (DTANSUG.F ``CONTITL``).
    configs : strong-field configurations with their B/C = 0.2 free-ion
        parentage (DTANSUG.F ``CONFIG``; obvious typos corrected, see
        docs/tanabe_sugano.md).
    fortran_line : line of the block's definition in DTANSUG.F.
    """

    nh: int
    first: int
    label: str
    terms: str
    configs: tuple[str, ...]
    fortran_line: int

    @property
    def size(self) -> int:
        return len(self.configs)

    @property
    def multiplicity(self) -> int:
        return int(self.label[0])

    @property
    def irrep(self) -> str:
        return self.label[1:]

    @property
    def irrep_dim(self) -> int:
        return {"A": 1, "E": 2, "T": 3}[self.irrep[0]]

    @property
    def degeneracy(self) -> int:
        """Spin x orbital degeneracy (2S+1) dim(Gamma) of each level."""
        return self.multiplicity * self.irrep_dim

    @property
    def last(self) -> int:
        return self.first + self.size - 1


def _B(nh, first, terms, configs, line):
    return Block(nh, first, configs[0].split()[0], terms, tuple(configs), line)


# DTANSUG.F ll. 381-749 (configuration labels).  Corrections relative to the
# Fortran strings are marked "# corrected".
BLOCKS: tuple[Block, ...] = (
    _B(1, 1, '2T2 (2D)',
       ('2T2 t (2D)',), 384),
    _B(1, 2, '2E  (2D)',
       ('2E  e (2D)',), 391),
    _B(2, 1, '1A1 (1G, 1S)',
       ('1A1 t2 (1G)', '1A1 e2 (1S)',), 400),
    _B(2, 3, '3A2 (3F)',
       ('3A2 e2 (3F)',), 408),
    _B(2, 4, '1E  (1D, 1G)',
       ('1E  t2 (1D)', '1E  e2 (1G)',), 415),
    _B(2, 6, '1T1 (1G)',
       ('1T1 et (1G)',), 423),
    _B(2, 7, '3T1 (3F, 3P)',
       ('3T1 t2 (3F)', '3T1 et (3P)',), 430),
    _B(2, 9, '1T2 (1D, 1G)',
       ('1T2 t2 (1D)', '1T2 et (1G)',), 438),
    _B(2, 11, '3T2 (3F)',
       ('3T2 et (3F)',), 446),
    _B(3, 1, '2T2 (2D(a), 2D(b), 2F, 2G, 2H)',
       ('2T2 t3       (2G)', '2T2 t2{3T1}e (2H)', '2T2 t2{1T2}e (2Da)', '2T2 te2{1A1} (2F)', '2T2 te2{1E}  (2Db)',), 455),
    _B(3, 6, '2T1 (2P, 2F, 2G, 2H)',
       ('2T1 t3       (2G)', '2T1 t2{3T1}e (2H)', '2T1 t2{1T2}e (2H)', '2T1 te2{3A2} (2P)', '2T1 te2{1E}  (2F)',), 466),
    _B(3, 11, '2E  (2D(a), 2D(b), 2G, 2H)',
       ('2E  t3       (2G)', '2E  t2{1A1}e (2H)', '2E  t2{1E}e  (2Da)', '2E  e3       (2Db)',), 477),
    _B(3, 15, '4T1 (4P, 4F)',
       ('4T1 t2{3T1}e (4P)', '4T1 te2{3A2} (4F)',), 487),
    _B(3, 17, '4A2 (4F)',
       ('4A2 t3       (4F)',), 495),
    _B(3, 18, '4T2 (4F)',
       ('4T2 t2{3T1}e (4F)',), 502),
    _B(3, 19, '2A1 (2G)',
       ('2A1 t2{1E}e  (2G)',), 509),
    _B(3, 20, '2A2 (2F)',
       ('2A2 t2{1E}e  (2F)',), 516),
    _B(4, 1, '1E  (1D(a), 1D(b), 1G(a), 1G(b), 1I)',
       ('1E  t4             (1I)', '1E  t3{2E}e        (1Ga)', '1E  t2{1E}e2{1A1}  (1Da)', '1E  t2{1A1}e2{1E}  (1Gb)', '1E  t2{1E}e2{1E}   (1Db)',), 525),
    _B(4, 6, '3T2 (3D, 3F(a), 3F(b), 3G, 3H)',
       ('3T2 t3{2T1}e       (3H)', '3T2 t3{2T2}e       (3Fa)', '3T2 t2{3T1}e2{3A2} (3G)', '3T2 t2{3T1}e2{1E}  (3D)', '3T2 te3            (3Fb)',), 536),
    _B(4, 11, '1T1 (1F, 1G(a), 1G(b), 1I)',
       ('1T1 t3{2T1}e       (1I)', '1T1 t3{2T2}e       (1Ga)', '1T1 t2{1T2}e2{1E}  (1F)', '1T1 te3            (1Gb)',), 547),
    _B(4, 15, '3E  (3D, 3G, 3H)',
       ('3E  t3{4A2}e       (3H)', '3E  t3{2E}e        (3G)', '3E  t2{1E}e2{3A2}  (3D)',), 557),
    _B(4, 18, '3A2 (3F(a), 3F(b))',
       ('3A2 t3{2E}e        (3Fa)', '3A2 t2{1A1}e2{3A2} (3Fb)',), 566),
    _B(4, 20, '1A2 (1F, 1I)',
       ('1A2 t3{2E}e        (1I)', '1A2 t2{1E}e2{1E}   (1F)',), 574),
    _B(4, 22, '5E  (5D)',
       ('5E  t3{4A2}e       (5D)',), 582),
    _B(4, 23, '5T2 (5D)',
       ('5T2 t2{3T1}e2{3A2} (5D)',), 589),
    _B(4, 24, '3A1 (3G)',
       ('3A1 t3{2E}e        (3G)',), 596),
    _B(4, 25, '3T1 (3P(a), 3P(b), 3F(a), 3F(b), 3G, 3H)',
       ('3T1 t4             (3H)', '3T1 t3{2T1}e       (3H)', '3T1 t3{2T2}e       (3Pa)', '3T1 t2{3T1}e2{1A1} (3Fa)', '3T1 t2{3T1}e2{1E}  (3G)', '3T1 t2{1T2}e2{3A2} (3Fb)', '3T1 te3            (3Pb)',), 603),
    _B(4, 32, '1T2 (1D(a), 1D(b), 1G(a), 1G(b), 1F, 1I)',
       ('1T2 t4             (1I)', '1T2 t3{2T1}e       (1I)', '1T2 t3{2T2}e       (1Ga)', '1T2 t2{3T1}e2{3A2} (1Da)', '1T2 t2{1T2}e2{1E}  (1F)', '1T2 t2{1T2}e2{1A1} (1Gb)', '1T2 te3            (1Db)',), 616),
    _B(4, 39, '1A1 (1S(a), 1S(b), 1G(a), 1G(b), 1I)',
       ('1A1 t4             (1I)', '1A1 t3{2E}e        (1Ga)', '1A1 t2{1A1}e2{1A1} (1Sa)', '1A1 t2{1E}e2{1E}   (1Gb)', '1A1 e4             (1Sb)',), 629),
    _B(5, 1, '2E  (2D(a), 2D(b), 2D(c), 2G(a), 2G(b), 2H, 2I)',
       ('2E  t4{1A1}e       (2I)', '2E  t4{1E}e        (2Da)', '2E  t3{2E}e2{1A1}  (2Ga)', '2E  t3{2E}e2{3A2}  (2H)', '2E  t3{2E}e2{1E}   (2Db)', '2E  t2{1E}e3       (2Gb)', '2E  t2{1A1}e3      (2Dc)',), 642),  # corrected
    _B(5, 8, '2A1 (2S, 2G(a), 2G(b), 2I)',
       ('2A1 t4{1E}e        (2I)', '2A1 t3{2E}e2{1E}   (2Ga)', '2A1 t3{4A2}e2{3A2} (2S)', '2A1 t2{1E}e3       (2Gb)',), 655),  # corrected
    _B(5, 12, '2A2 (2F(a), 2F(b), 2I)',
       ('2A2 t4{1E}e        (2I)', '2A2 t3{2E}e2{1E}   (2Fa)', '2A2 t2{1E}e3       (2Fb)',), 665),  # corrected
    _B(5, 15, '4T1 (4P, 4F, 4G)',
       ('4T1 t4{3T1}e       (4G)', '4T1 t3{2T2}e2{3A2} (4P)', '4T1 t2{3T1}e3      (4F)',), 674),
    _B(5, 18, '4T2 (4D, 4F, 4G)',
       ('4T2 t4{3T1}e       (4G)', '4T2 t3{2T1}e2{3A2} (4D)', '4T2 t2{3T1}e3      (4F)',), 683),
    _B(5, 21, '4E  (4D, 4G)',
       ('4E  t3{2E}e2{3A2}  (4G)', '4E  t3{4A2}e2{1E}  (4D)',), 692),  # corrected
    _B(5, 23, '6A1 (6S)',
       ('6A1 t3{4A2}e2{3A2} (6S)',), 700),
    _B(5, 24, '4A1 (4G)',
       ('4A1 t3{4A2}e2{3A2} (4G)',), 707),
    _B(5, 25, '4A2 (4F)',
       ('4A2 t3{4A2}e2{1A1} (4F)',), 714),
    _B(5, 26, '2T2 (2D(a,b,c), 2F(a), 2F(b), 2G(a), 2G(b), 2H, 2I)',
       ('2T2 t5             (2I)', '2T2 t4{3T1}e       (2I)', '2T2 t4{1T2}e       (2Da)', '2T2 t3{2T1}e2{3A2} (2Fa)', '2T2 t3{2T1}e2{1E}  (2Ga)', '2T2 t3{2T2}e2{1A1} (2H)', '2T2 t3{2T2}e2{1E}  (2Fb)', '2T2 t2{1T2}e3{2E}  (2Db)', '2T2 t2{3T1}e3{2E}  (2Gb)', '2T2 te4            (2Dc)',), 721),
    _B(5, 36, '2T1 (2P, 2F(a), 2F(b), 2G(a), 2G(b), 2H, 2I)',
       ('2T1 t4{3T1}e       (2I)', '2T1 t4{1T2}e       (2Fa)', '2T1 t3{2T1}e2{1A1} (2Ga)', '2T1 t3{2T1}e2{1E}  (2H)', '2T1 t3{2T2}e2{3A2} (2H)', '2T1 t3{2T2}e2{1E}  (2Fb)', '2T1 t2{1T2}e3      (2Gb)', '2T1 t2{3T1}e3      (2P)',), 737),  # corrected
)

_BY_KEY = {(b.nh, b.first): b for b in BLOCKS}

# Matrix elements (row, col, DTANSUG.F line, f(B, C, Dq, s)); upper triangle only.
# Dq here is the *effective* electron (or hole) cubic-field parameter and s is
# SIGNEL (+1 electrons, -1 holes); see hamiltonian().
_ELEMENTS: dict[tuple[int, int], tuple[Element, ...]] = {
    (1, 1): (
        (0, 0, 1411, lambda B, C, Dq, s: -4 * Dq),
    ),
    (1, 2): (
        (0, 0, 1415, lambda B, C, Dq, s: 6 * Dq),
    ),
    (2, 1): (
        (0, 0, 1422, lambda B, C, Dq, s: 10 * B + 5 * C - 8 * Dq),
        (1, 1, 1423, lambda B, C, Dq, s: 8 * B + 4 * C + 12 * Dq),
        (0, 1, 1424, lambda B, C, Dq, s: R6 * (2 * B + C)),
    ),
    (2, 3): (
        (0, 0, 1428, lambda B, C, Dq, s: -8 * B + 12 * Dq),
    ),
    (2, 4): (
        (0, 0, 1432, lambda B, C, Dq, s: B + 2 * C - 8 * Dq),
        (1, 1, 1433, lambda B, C, Dq, s: 2 * C + 12 * Dq),
        (0, 1, 1434, lambda B, C, Dq, s: s * 2 * R3 * B),
    ),
    (2, 6): (
        (0, 0, 1438, lambda B, C, Dq, s: 4 * B + 2 * C + 2 * Dq),
    ),
    (2, 7): (
        (0, 0, 1442, lambda B, C, Dq, s: -5 * B - 8 * Dq),
        (1, 1, 1443, lambda B, C, Dq, s: 4 * B + 2 * Dq),
        (0, 1, 1444, lambda B, C, Dq, s: s * (-6 * B)),
    ),
    (2, 9): (
        (0, 0, 1448, lambda B, C, Dq, s: B + 2 * C - 8 * Dq),
        (1, 1, 1449, lambda B, C, Dq, s: 2 * C + 2 * Dq),
        (0, 1, 1450, lambda B, C, Dq, s: s * (-2 * R3 * B)),
    ),
    (2, 11): (
        (0, 0, 1454, lambda B, C, Dq, s: -8 * B + 2 * Dq),
    ),
    (3, 1): (
        (0, 0, 1461, lambda B, C, Dq, s: 5 * C - 12 * Dq),
        (1, 1, 1462, lambda B, C, Dq, s: -6 * B + 3 * C - 2 * Dq),
        (2, 2, 1463, lambda B, C, Dq, s: 4 * B + 3 * C - 2 * Dq),
        (3, 3, 1464, lambda B, C, Dq, s: 6 * B + 5 * C + 8 * Dq),
        (4, 4, 1465, lambda B, C, Dq, s: -2 * B + 3 * C + 8 * Dq),
        (0, 1, 1466, lambda B, C, Dq, s: -3 * R3 * B),
        (0, 2, 1467, lambda B, C, Dq, s: s * (5 * R3 * B)),
        (0, 3, 1468, lambda B, C, Dq, s: 4 * B + 2 * C),
        (0, 4, 1469, lambda B, C, Dq, s: 2 * B),
        (1, 2, 1470, lambda B, C, Dq, s: s * (-3 * B)),
        (1, 3, 1471, lambda B, C, Dq, s: -3 * R3 * B),
        (1, 4, 1472, lambda B, C, Dq, s: -3 * R3 * B),
        (2, 3, 1473, lambda B, C, Dq, s: s * (R3 * B)),
        (2, 4, 1474, lambda B, C, Dq, s: s * (-R3 * B)),
        (3, 4, 1475, lambda B, C, Dq, s: 10 * B),
    ),
    (3, 6): (
        (0, 0, 1479, lambda B, C, Dq, s: -6 * B + 3 * C - 12 * Dq),
        (1, 1, 1480, lambda B, C, Dq, s: 3 * C - 2 * Dq),
        (2, 2, 1481, lambda B, C, Dq, s: -6 * B + 3 * C - 2 * Dq),
        (3, 3, 1482, lambda B, C, Dq, s: -6 * B + 3 * C + 8 * Dq),
        (4, 4, 1483, lambda B, C, Dq, s: -2 * B + 3 * C + 8 * Dq),
        (0, 1, 1484, lambda B, C, Dq, s: s * (3 * B)),
        (0, 2, 1485, lambda B, C, Dq, s: s * (-3 * B)),
        (0, 3, 1486, lambda B, C, Dq, s: 0),
        (0, 4, 1487, lambda B, C, Dq, s: -2 * R3 * B),
        (1, 2, 1488, lambda B, C, Dq, s: -3 * B),
        (1, 3, 1489, lambda B, C, Dq, s: 3 * B),
        (1, 4, 1490, lambda B, C, Dq, s: s * (-3 * R3 * B)),
        (2, 3, 1491, lambda B, C, Dq, s: -3 * B),
        (2, 4, 1492, lambda B, C, Dq, s: s * (R3 * B)),
        (3, 4, 1493, lambda B, C, Dq, s: s * (-2 * R3 * B)),
    ),
    (3, 11): (
        (0, 0, 1497, lambda B, C, Dq, s: -6 * B + 3 * C - 12 * Dq),
        (1, 1, 1498, lambda B, C, Dq, s: 8 * B + 6 * C - 2 * Dq),
        (2, 2, 1499, lambda B, C, Dq, s: -1 * B + 3 * C - 2 * Dq),
        (3, 3, 1500, lambda B, C, Dq, s: -8 * B + 4 * C + 18 * Dq),
        (0, 1, 1501, lambda B, C, Dq, s: -6 * R2 * B),
        (0, 2, 1502, lambda B, C, Dq, s: s * (3 * R2 * B)),
        (0, 3, 1503, lambda B, C, Dq, s: 0),
        (1, 2, 1504, lambda B, C, Dq, s: s * (-10 * B)),
        (1, 3, 1505, lambda B, C, Dq, s: R3 * (2 * B + C)),
        (2, 3, 1506, lambda B, C, Dq, s: s * (-2 * R3 * B)),
    ),
    (3, 15): (
        (0, 0, 1510, lambda B, C, Dq, s: -3 * B - 2 * Dq),
        (1, 1, 1511, lambda B, C, Dq, s: -12 * B + 8 * Dq),
        (0, 1, 1512, lambda B, C, Dq, s: 6 * B),
    ),
    (3, 17): (
        (0, 0, 1516, lambda B, C, Dq, s: -15 * B - 12 * Dq),
    ),
    (3, 18): (
        (0, 0, 1520, lambda B, C, Dq, s: -15 * B - 2 * Dq),
    ),
    (3, 19): (
        (0, 0, 1524, lambda B, C, Dq, s: -11 * B + 3 * C - 2 * Dq),
    ),
    (3, 20): (
        (0, 0, 1528, lambda B, C, Dq, s: 9 * B + 3 * C - 2 * Dq),
    ),
    (4, 1): (
        (0, 0, 1535, lambda B, C, Dq, s: -9 * B + 7 * C - 16 * Dq),
        (1, 1, 1536, lambda B, C, Dq, s: -6 * B + 6 * C - 6 * Dq),
        (2, 2, 1537, lambda B, C, Dq, s: 5 * B + 8 * C + 4 * Dq),
        (3, 3, 1538, lambda B, C, Dq, s: 6 * B + 9 * C + 4 * Dq),
        (4, 4, 1539, lambda B, C, Dq, s: -3 * B + 6 * C + 4 * Dq),
        (0, 1, 1540, lambda B, C, Dq, s: 6 * B),
        (0, 2, 1541, lambda B, C, Dq, s: s * (-R2 * (2 * B + C))),
        (0, 3, 1542, lambda B, C, Dq, s: -2 * B),
        (0, 4, 1543, lambda B, C, Dq, s: s * (4 * B)),
        (1, 2, 1544, lambda B, C, Dq, s: s * (3 * R2 * B)),
        (1, 3, 1545, lambda B, C, Dq, s: -12 * B),
        (1, 4, 1546, lambda B, C, Dq, s: 0),
        (2, 3, 1547, lambda B, C, Dq, s: s * (-10 * R2 * B)),
        (2, 4, 1548, lambda B, C, Dq, s: -10 * R2 * B),
        (3, 4, 1549, lambda B, C, Dq, s: 0),
    ),
    (4, 6): (
        (0, 0, 1553, lambda B, C, Dq, s: -9 * B + 4 * C - 6 * Dq),
        (1, 1, 1554, lambda B, C, Dq, s: -5 * B + 6 * C - 6 * Dq),
        (2, 2, 1555, lambda B, C, Dq, s: -13 * B + 4 * C + 4 * Dq),
        (3, 3, 1556, lambda B, C, Dq, s: -9 * B + 4 * C + 4 * Dq),
        (4, 4, 1557, lambda B, C, Dq, s: -8 * B + 5 * C + 14 * Dq),
        (0, 1, 1558, lambda B, C, Dq, s: s * (5 * R3 * B)),
        (0, 2, 1559, lambda B, C, Dq, s: R6 * B),
        (0, 3, 1560, lambda B, C, Dq, s: s * (-R3 * B)),
        (0, 4, 1561, lambda B, C, Dq, s: s * (R6 * B)),
        (1, 2, 1562, lambda B, C, Dq, s: s * (3 * R2 * B)),
        (1, 3, 1563, lambda B, C, Dq, s: 3 * B),
        (1, 4, 1564, lambda B, C, Dq, s: R2 * (3 * B + C)),
        (2, 3, 1565, lambda B, C, Dq, s: s * (2 * R2 * B)),
        (2, 4, 1566, lambda B, C, Dq, s: s * (6 * B)),
        (3, 4, 1567, lambda B, C, Dq, s: 3 * R2 * B),
    ),
    (4, 11): (
        (0, 0, 1571, lambda B, C, Dq, s: -3 * B + 6 * C - 6 * Dq),
        (1, 1, 1572, lambda B, C, Dq, s: -3 * B + 8 * C - 6 * Dq),
        (2, 2, 1573, lambda B, C, Dq, s: -3 * B + 6 * C + 4 * Dq),
        (3, 3, 1574, lambda B, C, Dq, s: -16 * B + 7 * C + 14 * Dq),
        (0, 1, 1575, lambda B, C, Dq, s: 5 * R3 * B),
        (0, 2, 1576, lambda B, C, Dq, s: s * (-3 * B)),
        (0, 3, 1577, lambda B, C, Dq, s: R6 * B),
        (1, 2, 1578, lambda B, C, Dq, s: s * (5 * R3 * B)),
        (1, 3, 1579, lambda B, C, Dq, s: R2 * (B + C)),
        (2, 3, 1580, lambda B, C, Dq, s: s * (R6 * B)),
    ),
    (4, 15): (
        (0, 0, 1584, lambda B, C, Dq, s: -13 * B + 4 * C - 6 * Dq),
        (1, 1, 1585, lambda B, C, Dq, s: -10 * B + 4 * C - 6 * Dq),
        (2, 2, 1586, lambda B, C, Dq, s: -11 * B + 4 * C + 4 * Dq),
        (0, 1, 1587, lambda B, C, Dq, s: -4 * B),
        (0, 2, 1588, lambda B, C, Dq, s: 0),
        (1, 2, 1589, lambda B, C, Dq, s: s * (3 * R2 * B)),
    ),
    (4, 18): (
        (0, 0, 1593, lambda B, C, Dq, s: -8 * B + 4 * C - 6 * Dq),
        (1, 1, 1594, lambda B, C, Dq, s: -2 * B + 7 * C + 4 * Dq),
        (0, 1, 1595, lambda B, C, Dq, s: -12 * B),
    ),
    (4, 20): (
        (0, 0, 1599, lambda B, C, Dq, s: -12 * B + 6 * C - 6 * Dq),
        (1, 1, 1600, lambda B, C, Dq, s: -3 * B + 6 * C + 4 * Dq),
        (0, 1, 1601, lambda B, C, Dq, s: s * (-6 * B)),
    ),
    (4, 22): (
        (0, 0, 1605, lambda B, C, Dq, s: -21 * B - 6 * Dq),
    ),
    (4, 23): (
        (0, 0, 1609, lambda B, C, Dq, s: -21 * B + 4 * Dq),
    ),
    (4, 24): (
        (0, 0, 1613, lambda B, C, Dq, s: -12 * B + 4 * C - 6 * Dq),
    ),
    (4, 25): (
        (0, 0, 1617, lambda B, C, Dq, s: -15 * B + 5 * C - 16 * Dq),
        (1, 1, 1618, lambda B, C, Dq, s: -11 * B + 4 * C - 6 * Dq),
        (2, 2, 1619, lambda B, C, Dq, s: -3 * B + 6 * C - 6 * Dq),
        (3, 3, 1620, lambda B, C, Dq, s: -1 * B + 6 * C + 4 * Dq),
        (4, 4, 1621, lambda B, C, Dq, s: -9 * B + 4 * C + 4 * Dq),
        (5, 5, 1622, lambda B, C, Dq, s: -11 * B + 4 * C + 4 * Dq),
        (6, 6, 1623, lambda B, C, Dq, s: -16 * B + 5 * C + 14 * Dq),
        (0, 1, 1624, lambda B, C, Dq, s: -R6 * B),
        (0, 2, 1625, lambda B, C, Dq, s: -3 * R2 * B),
        (0, 3, 1626, lambda B, C, Dq, s: s * (-R2 * (2 * B + C))),
        (0, 4, 1627, lambda B, C, Dq, s: s * (2 * R2 * B)),
        (0, 5, 1628, lambda B, C, Dq, s: 0),
        (0, 6, 1629, lambda B, C, Dq, s: 0),
        (1, 2, 1630, lambda B, C, Dq, s: 5 * R3 * B),
        (1, 3, 1631, lambda B, C, Dq, s: s * (-R3 * B)),
        (1, 4, 1632, lambda B, C, Dq, s: s * (R3 * B)),
        (1, 5, 1633, lambda B, C, Dq, s: 3 * B),
        (1, 6, 1634, lambda B, C, Dq, s: R6 * B),
        (2, 3, 1635, lambda B, C, Dq, s: s * (3 * B)),
        (2, 4, 1636, lambda B, C, Dq, s: s * (3 * B)),
        (2, 5, 1637, lambda B, C, Dq, s: 5 * R3 * B),
        (2, 6, 1638, lambda B, C, Dq, s: R2 * (B + C)),
        (3, 4, 1639, lambda B, C, Dq, s: -10 * B),
        (3, 5, 1640, lambda B, C, Dq, s: 0),
        (3, 6, 1641, lambda B, C, Dq, s: s * (-3 * R2 * B)),
        (4, 5, 1642, lambda B, C, Dq, s: s * (2 * R3 * B)),
        (4, 6, 1643, lambda B, C, Dq, s: s * (3 * R2 * B)),
        (5, 6, 1644, lambda B, C, Dq, s: R6 * B),
    ),
    (4, 32): (
        (0, 0, 1648, lambda B, C, Dq, s: -9 * B + 7 * C - 16 * Dq),
        (1, 1, 1649, lambda B, C, Dq, s: -9 * B + 6 * C - 6 * Dq),
        (2, 2, 1650, lambda B, C, Dq, s: 3 * B + 8 * C - 6 * Dq),
        (3, 3, 1651, lambda B, C, Dq, s: -9 * B + 6 * C + 4 * Dq),
        (4, 4, 1652, lambda B, C, Dq, s: -3 * B + 6 * C + 4 * Dq),
        (5, 5, 1653, lambda B, C, Dq, s: 5 * B + 8 * C + 4 * Dq),
        (6, 6, 1654, lambda B, C, Dq, s: 7 * C + 14 * Dq),
        (0, 1, 1655, lambda B, C, Dq, s: s * (-3 * R2 * B)),
        (0, 2, 1656, lambda B, C, Dq, s: -5 * R6 * B),
        (0, 3, 1657, lambda B, C, Dq, s: 0),
        (0, 4, 1658, lambda B, C, Dq, s: s * (2 * R2 * B)),
        (0, 5, 1659, lambda B, C, Dq, s: s * (-R2 * (2 * B + C))),
        (0, 6, 1660, lambda B, C, Dq, s: 0),
        (1, 2, 1661, lambda B, C, Dq, s: s * (5 * R3 * B)),
        (1, 3, 1662, lambda B, C, Dq, s: 3 * B),
        (1, 4, 1663, lambda B, C, Dq, s: -3 * B),
        (1, 5, 1664, lambda B, C, Dq, s: -3 * B),
        (1, 6, 1665, lambda B, C, Dq, s: s * (R6 * B)),
        (2, 3, 1666, lambda B, C, Dq, s: s * (3 * R3 * B)),
        (2, 4, 1667, lambda B, C, Dq, s: s * (-5 * R3 * B)),
        (2, 5, 1668, lambda B, C, Dq, s: s * (5 * R3 * B)),
        (2, 6, 1669, lambda B, C, Dq, s: R2 * (3 * B + C)),
        (3, 4, 1670, lambda B, C, Dq, s: -6 * B),
        (3, 5, 1671, lambda B, C, Dq, s: 0),
        (3, 6, 1672, lambda B, C, Dq, s: s * (3 * R6 * B)),
        (4, 5, 1673, lambda B, C, Dq, s: -10 * B),
        (4, 6, 1674, lambda B, C, Dq, s: s * (-R6 * B)),
        (5, 6, 1675, lambda B, C, Dq, s: s * (-R6 * B)),
    ),
    (4, 39): (
        (0, 0, 1679, lambda B, C, Dq, s: 10 * C - 16 * Dq),
        (1, 1, 1680, lambda B, C, Dq, s: 6 * C - 6 * Dq),
        (2, 2, 1681, lambda B, C, Dq, s: 14 * B + 11 * C + 4 * Dq),
        (3, 3, 1682, lambda B, C, Dq, s: -3 * B + 6 * C + 4 * Dq),
        (4, 4, 1683, lambda B, C, Dq, s: -16 * B + 8 * C + 24 * Dq),
        (0, 1, 1684, lambda B, C, Dq, s: -12 * R2 * B),
        (0, 2, 1685, lambda B, C, Dq, s: R2 * (4 * B + 2 * C)),
        (0, 3, 1686, lambda B, C, Dq, s: s * (-2 * R2 * B)),
        (0, 4, 1687, lambda B, C, Dq, s: 0),
        (1, 2, 1688, lambda B, C, Dq, s: -12 * B),
        (1, 3, 1689, lambda B, C, Dq, s: s * (6 * B)),
        (1, 4, 1690, lambda B, C, Dq, s: 0),
        (2, 3, 1691, lambda B, C, Dq, s: s * (-20 * B)),
        (2, 4, 1692, lambda B, C, Dq, s: R6 * (2 * B + C)),
        (3, 4, 1693, lambda B, C, Dq, s: s * (-2 * R6 * B)),
    ),
    (5, 1): (
        (0, 0, 1699, lambda B, C, Dq, s: -4 * B + 12 * C - 10 * Dq),
        (1, 1, 1700, lambda B, C, Dq, s: -13 * B + 9 * C - 10 * Dq),
        (2, 2, 1701, lambda B, C, Dq, s: -4 * B + 10 * C),
        (3, 3, 1702, lambda B, C, Dq, s: -16 * B + 8 * C),
        (4, 4, 1703, lambda B, C, Dq, s: -12 * B + 8 * C),
        (5, 5, 1704, lambda B, C, Dq, s: -13 * B + 9 * C + 10 * Dq),
        (6, 6, 1705, lambda B, C, Dq, s: -4 * B + 12 * C + 10 * Dq),
        (0, 1, 1706, lambda B, C, Dq, s: 10 * B),
        (0, 2, 1707, lambda B, C, Dq, s: 6 * B),
        (0, 3, 1708, lambda B, C, Dq, s: 6 * R3 * B),
        (0, 4, 1709, lambda B, C, Dq, s: 6 * R2 * B),
        (0, 5, 1710, lambda B, C, Dq, s: -2 * B),
        (0, 6, 1711, lambda B, C, Dq, s: 4 * B + 2 * C),
        (1, 2, 1712, lambda B, C, Dq, s: -3 * B),
        (1, 3, 1713, lambda B, C, Dq, s: 3 * R3 * B),
        (1, 4, 1714, lambda B, C, Dq, s: 0),
        (1, 5, 1715, lambda B, C, Dq, s: 2 * B + C),
        (1, 6, 1716, lambda B, C, Dq, s: 2 * B),
        (2, 3, 1717, lambda B, C, Dq, s: 0),
        (2, 4, 1718, lambda B, C, Dq, s: 0),
        (2, 5, 1719, lambda B, C, Dq, s: -3 * B),
        (2, 6, 1720, lambda B, C, Dq, s: -6 * B),
        (3, 4, 1721, lambda B, C, Dq, s: 2 * R6 * B),
        (3, 5, 1722, lambda B, C, Dq, s: -3 * R3 * B),
        (3, 6, 1723, lambda B, C, Dq, s: 6 * R3 * B),
        (4, 5, 1724, lambda B, C, Dq, s: 0),
        (4, 6, 1725, lambda B, C, Dq, s: 6 * R2 * B),
        (5, 6, 1726, lambda B, C, Dq, s: -10 * B),
    ),
    (5, 8): (
        (0, 0, 1730, lambda B, C, Dq, s: -3 * B + 9 * C - 10 * Dq),
        (1, 1, 1731, lambda B, C, Dq, s: -12 * B + 8 * C),
        (2, 2, 1732, lambda B, C, Dq, s: -19 * B + 8 * C),
        (3, 3, 1733, lambda B, C, Dq, s: -3 * B + 9 * C + 10 * Dq),
        (0, 1, 1734, lambda B, C, Dq, s: -3 * R2 * B),
        (0, 2, 1735, lambda B, C, Dq, s: 0),
        (0, 3, 1736, lambda B, C, Dq, s: 6 * B + C),
        (1, 2, 1737, lambda B, C, Dq, s: -4 * R3 * B),
        (1, 3, 1738, lambda B, C, Dq, s: 3 * R2 * B),
        (2, 3, 1739, lambda B, C, Dq, s: 0),
    ),
    (5, 12): (
        (0, 0, 1743, lambda B, C, Dq, s: -23 * B + 9 * C - 10 * Dq),
        (1, 1, 1744, lambda B, C, Dq, s: -12 * B + 8 * C),
        (2, 2, 1745, lambda B, C, Dq, s: -23 * B + 9 * C + 10 * Dq),
        (0, 1, 1746, lambda B, C, Dq, s: 3 * R2 * B),
        (0, 2, 1747, lambda B, C, Dq, s: -2 * B + C),
        (1, 2, 1748, lambda B, C, Dq, s: -3 * R2 * B),
    ),
    (5, 15): (
        (0, 0, 1752, lambda B, C, Dq, s: -25 * B + 6 * C - 10 * Dq),
        (1, 1, 1753, lambda B, C, Dq, s: -16 * B + 7 * C),
        (2, 2, 1754, lambda B, C, Dq, s: -25 * B + 6 * C + 10 * Dq),
        (0, 1, 1755, lambda B, C, Dq, s: -3 * R2 * B),
        (0, 2, 1756, lambda B, C, Dq, s: C),
        (1, 2, 1757, lambda B, C, Dq, s: -3 * R2 * B),
    ),
    (5, 18): (
        (0, 0, 1761, lambda B, C, Dq, s: -17 * B + 6 * C - 10 * Dq),
        (1, 1, 1762, lambda B, C, Dq, s: -22 * B + 5 * C),
        (2, 2, 1763, lambda B, C, Dq, s: -17 * B + 6 * C + 10 * Dq),
        (0, 1, 1764, lambda B, C, Dq, s: R6 * B),
        (0, 2, 1765, lambda B, C, Dq, s: 4 * B + C),
        (1, 2, 1766, lambda B, C, Dq, s: -R6 * B),
    ),
    (5, 21): (
        (0, 0, 1770, lambda B, C, Dq, s: -22 * B + 5 * C),
        (1, 1, 1771, lambda B, C, Dq, s: -21 * B + 5 * C),
        (0, 1, 1772, lambda B, C, Dq, s: -2 * R3 * B),
    ),
    (5, 23): (
        (0, 0, 1776, lambda B, C, Dq, s: -35 * B),
    ),
    (5, 24): (
        (0, 0, 1780, lambda B, C, Dq, s: -25 * B + 5 * C),
    ),
    (5, 25): (
        (0, 0, 1784, lambda B, C, Dq, s: -13 * B + 7 * C),
    ),
    (5, 26): (
        (0, 0, 1788, lambda B, C, Dq, s: -20 * B + 10 * C - 20 * Dq),
        (1, 1, 1789, lambda B, C, Dq, s: -8 * B + 9 * C - 10 * Dq),
        (2, 2, 1790, lambda B, C, Dq, s: -18 * B + 9 * C - 10 * Dq),
        (3, 3, 1791, lambda B, C, Dq, s: -16 * B + 8 * C),
        (4, 4, 1792, lambda B, C, Dq, s: -12 * B + 8 * C),
        (5, 5, 1793, lambda B, C, Dq, s: 2 * B + 12 * C),
        (6, 6, 1794, lambda B, C, Dq, s: -6 * B + 10 * C),
        (7, 7, 1795, lambda B, C, Dq, s: -18 * B + 9 * C + 10 * Dq),
        (8, 8, 1796, lambda B, C, Dq, s: -8 * B + 9 * C + 10 * Dq),
        (9, 9, 1797, lambda B, C, Dq, s: -20 * B + 10 * C + 20 * Dq),
        (0, 1, 1798, lambda B, C, Dq, s: 3 * R6 * B),
        (0, 2, 1799, lambda B, C, Dq, s: R6 * B),
        (0, 3, 1800, lambda B, C, Dq, s: 0),
        (0, 4, 1801, lambda B, C, Dq, s: -2 * R3 * B),
        (0, 5, 1802, lambda B, C, Dq, s: 4 * B + 2 * C),
        (0, 6, 1803, lambda B, C, Dq, s: 2 * B),
        (0, 7, 1804, lambda B, C, Dq, s: 0),
        (0, 8, 1805, lambda B, C, Dq, s: 0),
        (0, 9, 1806, lambda B, C, Dq, s: 0),
        (1, 2, 1807, lambda B, C, Dq, s: 3 * B),
        (1, 3, 1808, lambda B, C, Dq, s: (R6/2) * B),
        (1, 4, 1809, lambda B, C, Dq, s: -(3/R2) * B),
        (1, 5, 1810, lambda B, C, Dq, s: 3 * (R6/2) * B),
        (1, 6, 1811, lambda B, C, Dq, s: 3 * (R6/2) * B),
        (1, 7, 1812, lambda B, C, Dq, s: 0),
        (1, 8, 1813, lambda B, C, Dq, s: 4 * B + C),
        (1, 9, 1814, lambda B, C, Dq, s: 0 * B),
        (2, 3, 1815, lambda B, C, Dq, s: 3 * (R6/2) * B),
        (2, 4, 1816, lambda B, C, Dq, s: -(3/R2) * B),
        (2, 5, 1817, lambda B, C, Dq, s: 5 * (R6/2) * B),
        (2, 6, 1818, lambda B, C, Dq, s: -5 * (R6/2) * B),
        (2, 7, 1819, lambda B, C, Dq, s: C),
        (2, 8, 1820, lambda B, C, Dq, s: 0),
        (2, 9, 1821, lambda B, C, Dq, s: 0),
        (3, 4, 1822, lambda B, C, Dq, s: 2 * R3 * B),
        (3, 5, 1823, lambda B, C, Dq, s: 0),
        (3, 6, 1824, lambda B, C, Dq, s: 0),
        (3, 7, 1825, lambda B, C, Dq, s: -3 * (R6/2) * B),
        (3, 8, 1826, lambda B, C, Dq, s: -(R6/2) * B),
        (3, 9, 1827, lambda B, C, Dq, s: 0),
        (4, 5, 1828, lambda B, C, Dq, s: -10 * R3 * B),
        (4, 6, 1829, lambda B, C, Dq, s: 0),
        (4, 7, 1830, lambda B, C, Dq, s: (3/R2) * B),
        (4, 8, 1831, lambda B, C, Dq, s: (3/R2) * B),
        (4, 9, 1832, lambda B, C, Dq, s: -2 * R3 * B),
        (5, 6, 1833, lambda B, C, Dq, s: 0),
        (5, 7, 1834, lambda B, C, Dq, s: -5 * (R6/2) * B),
        (5, 8, 1835, lambda B, C, Dq, s: -3 * (R6/2) * B),
        (5, 9, 1836, lambda B, C, Dq, s: 4 * B + 2 * C),
        (6, 7, 1837, lambda B, C, Dq, s: -5 * (R6/2) * B),
        (6, 8, 1838, lambda B, C, Dq, s: 3 * (R6/2) * B),
        (6, 9, 1839, lambda B, C, Dq, s: -2 * B),
        (7, 8, 1840, lambda B, C, Dq, s: 3 * B),
        (7, 9, 1841, lambda B, C, Dq, s: -R6 * B),
        (8, 9, 1842, lambda B, C, Dq, s: -3 * R6 * B),
    ),
    (5, 36): (
        (0, 0, 1846, lambda B, C, Dq, s: -22 * B + 9 * C - 10 * Dq),
        (1, 1, 1847, lambda B, C, Dq, s: -8 * B + 9 * C - 10 * Dq),
        (2, 2, 1848, lambda B, C, Dq, s: -4 * B + 10 * C),
        (3, 3, 1849, lambda B, C, Dq, s: -12 * B + 8 * C),
        (4, 4, 1850, lambda B, C, Dq, s: -10 * B + 10 * C),
        (5, 5, 1851, lambda B, C, Dq, s: -6 * B + 10 * C),
        (6, 6, 1852, lambda B, C, Dq, s: -8 * B + 9 * C + 10 * Dq),
        (7, 7, 1853, lambda B, C, Dq, s: -22 * B + 9 * C + 10 * Dq),
        (0, 1, 1854, lambda B, C, Dq, s: -3 * B),
        (0, 2, 1855, lambda B, C, Dq, s: -(3/R2) * B),
        (0, 3, 1856, lambda B, C, Dq, s: (3/R2) * B),
        (0, 4, 1857, lambda B, C, Dq, s: -(3/R2) * B),
        (0, 5, 1858, lambda B, C, Dq, s: -3 * (R6/2) * B),
        (0, 6, 1859, lambda B, C, Dq, s: 0),
        (0, 7, 1860, lambda B, C, Dq, s: C),
        (1, 2, 1861, lambda B, C, Dq, s: (3/R2) * B),
        (1, 3, 1862, lambda B, C, Dq, s: (3/R2) * B),
        (1, 4, 1863, lambda B, C, Dq, s: (15/R2) * B),
        (1, 5, 1864, lambda B, C, Dq, s: 5 * (R6/2) * B),
        (1, 6, 1865, lambda B, C, Dq, s: 4 * B + C),
        (1, 7, 1866, lambda B, C, Dq, s: 0),
        (2, 3, 1867, lambda B, C, Dq, s: 0),
        (2, 4, 1868, lambda B, C, Dq, s: 0),
        (2, 5, 1869, lambda B, C, Dq, s: 10 * R3 * B),
        (2, 6, 1870, lambda B, C, Dq, s: (3/R2) * B),
        (2, 7, 1871, lambda B, C, Dq, s: -(3/R2) * B),
        (3, 4, 1872, lambda B, C, Dq, s: 0),
        (3, 5, 1873, lambda B, C, Dq, s: 0),
        (3, 6, 1874, lambda B, C, Dq, s: -(3/R2) * B),
        (3, 7, 1875, lambda B, C, Dq, s: -(3/R2) * B),
        (4, 5, 1876, lambda B, C, Dq, s: 2 * R3 * B),
        (4, 6, 1877, lambda B, C, Dq, s: (15/R2) * B),
        (4, 7, 1878, lambda B, C, Dq, s: -(3/R2) * B),
        (5, 6, 1879, lambda B, C, Dq, s: 5 * (R6/2) * B),
        (5, 7, 1880, lambda B, C, Dq, s: -3 * (R6/2) * B),
        (6, 7, 1881, lambda B, C, Dq, s: -3 * B),
    ),
}


def _check_n(n: int) -> int:
    if not 1 <= int(n) <= 9 or int(n) != n:
        raise ValueError(f"number of d electrons must be 1..9, got {n!r}")
    return int(n)


def _nh(n: int) -> int:
    """Electrons (n <= 5) or holes (n > 5) actually treated (DTANSUG.F ll. 77-82)."""
    n = _check_n(n)
    return n if n <= 5 else 10 - n


def _check_geometry(geometry: str) -> str:
    g = {"oh": "Oh", "o": "Oh", "td": "Td", "t": "Td"}.get(str(geometry).lower())
    if g is None:
        raise ValueError(f"geometry must be 'Oh' or 'Td', got {geometry!r}")
    return g


def blocks_for(n: int) -> tuple[Block, ...]:
    """All blocks of d^n, in DTANSUG.F menu order (equal for d^n and d^(10-n))."""
    nh = _nh(n)
    return tuple(b for b in BLOCKS if b.nh == nh)


def get_block(n: int, label: str) -> Block:
    """Block of d^n by its label, e.g. ``get_block(3, "4T2")``."""
    lab = label.replace(" ", "").upper()
    for b in blocks_for(n):
        if b.label.upper() == lab:
            return b
    have = ", ".join(b.label for b in blocks_for(n))
    raise KeyError(f"d{n} has no {label!r} block; blocks: {have}")


def block_for_nfig(n: int, nfig: int) -> Block:
    """Block selected by the DTANSUG.F menu number NFIG (ll. 309-376)."""
    for b in blocks_for(n):
        if b.first <= nfig <= b.last:
            return b
    raise KeyError(f"NFIG={nfig} is not a valid menu entry for d{n}")


def electron_hole_sign(n: int) -> float:
    """``SIGNEL``: +1 for d1-d5, -1 for d6-d9 (DTANSUG.F ll. 46, 80)."""
    return 1.0 if _check_n(n) <= 5 else -1.0


def cubic_sign(n: int, geometry: str = "Oh") -> float:
    """``SIGNXF``: sign multiplying the user's Dq (DTANSUG.F ll. 45, 79, 86-87).

    -1 for holes (n > 5), times -1 again for tetrahedral symmetry.
    """
    s = electron_hole_sign(n)
    return -s if _check_geometry(geometry) == "Td" else s


def diag_offset(n: int, B: float, C: float) -> float:
    """``DIAG``: constant added to every diagonal element for d6, d7, d8.

    Puts the d^(10-n) hole energies on the d^n scale (A term excluded):
    d8: -42B + 21C (l. 1420); d7: -28B + 14C (l. 1459); d6: -14B + 7C (l. 1533).
    DTANSUG.F adds no constant for d9 (its two levels are just +4Dq, -6Dq).
    """
    return {8: -42.0 * B + 21.0 * C, 7: -28.0 * B + 14.0 * C, 6: -14.0 * B + 7.0 * C}.get(_check_n(n), 0.0)


def hamiltonian(n: int, label: str | Block, B: float, C: float, Dq: float, geometry: str = "Oh") -> np.ndarray:
    """Real symmetric strong-field matrix of one block of d^n (cm-1).

    Parameters
    ----------
    n : number of d electrons, 1-9.
    label : block label such as ``"3T1"`` (or a :class:`Block`).
    B, C : Racah parameters.
    Dq : cubic-field parameter of the given geometry (Dq > 0 is the normal
        physical sign: octahedral e(g) above t2(g), tetrahedral e below t2).
    geometry : ``"Oh"`` or ``"Td"``.

    Returns
    -------
    (size, size) array H with H_ij = <config i|H|config j>, configurations in
    ``Block.configs`` order. Equals DTANSUG.F's ``HR`` after l. 1894 for the
    Dq loop, i.e. with the effective field ``SIGNXF * Dq`` (l. 1375).
    """
    blk = label if isinstance(label, Block) else get_block(n, label)
    if blk.nh != _nh(n):
        raise ValueError(f"block {blk.label} (nh={blk.nh}) does not belong to d{n}")
    dq_eff = cubic_sign(n, geometry) * Dq
    s = electron_hole_sign(n)
    h = np.zeros((blk.size, blk.size))
    for i, j, _line, f in _ELEMENTS[(blk.nh, blk.first)]:
        h[i, j] = f(B, C, dq_eff, s)
        h[j, i] = h[i, j]
    h[np.diag_indices(blk.size)] += diag_offset(n, B, C)
    return h
