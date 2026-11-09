"""Clementi-Raimondi effective nuclear charge of the metal d shell (copied from telser-tools).

Verbatim copy of ``telser_tools.dipole.zeff`` (dipole.f subroutine CLEMRAIM, l. 388-425), used
by :mod:`torchspin.lft.dipole`; ``torchspin/tests/test_lft_dipole.py`` checks the copy against
telser-tools when it is installed.

* Z <= 20:            Z* = Z (no screening; no d electrons exist there anyway)        l. 395-398
* Z > 20 (generic):   Z* = Z - [13.5894 + 0.2693 (n_d - 1)]                           l. 399
  This is the Clementi-Raimondi screening of a **3d** electron by the [Ar] core plus the other
  (n_d - 1) 3d electrons (E. Clementi, D. L. Raimondi, J. Chem. Phys. 1963, 38, 2686, with no
  4s/4p electrons).  It is meaningless for 4d/5d/4f shells.
* Z = 21-30 and 39-48: replaced by tabulated neutral-atom 3d / 4d values (E. Clementi,
  D. L. Raimondi, W. P. Reinhardt, J. Chem. Phys. 1967, 47, 1300; as listed on Wikipedia,
  "Effective nuclear charge", per the Fortran comment l. 401).  n_d is ignored there.     l. 404-423
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ZStar", "clementi_raimondi", "TABULATED"]

TABULATED = {
    21: 7.120, 22: 8.141, 23: 8.983, 24: 9.757, 25: 10.528, 26: 11.180, 27: 11.855, 28: 12.530,
    29: 13.201, 30: 13.878,
    39: 15.958, 40: 13.072, 41: 11.238, 42: 11.392, 43: 12.882, 44: 12.813, 45: 13.442, 46: 13.618,
    47: 14.763, 48: 15.877,
}


@dataclass(frozen=True)
class ZStar:
    value: float
    method: str                      # 'unscreened', 'tabulated 3d', 'tabulated 4d', 'generic 3d formula'
    notes: tuple[str, ...] = field(default=())


def clementi_raimondi(n_d: int, z: int) -> ZStar:
    """Z* exactly as CLEMRAIM computes it, with notes on where the model does not apply."""
    if z <= 20:
        return ZStar(float(z), "unscreened", ("Z <= 20: CLEMRAIM returns Z* = Z (no screening).",))
    if z in TABULATED:
        if z <= 30:
            return ZStar(TABULATED[z], "tabulated 3d")
        return ZStar(TABULATED[z], "tabulated 4d", (
            "4d metal: Z* is for a 4d (n = 4) orbital, but the dipolar tensor uses a 3d (n = 3) "
            "Slater radial function, as dipole.f does.",))
    n_dm1 = max(n_d - 1, 0)
    notes = ["Generic Clementi-Raimondi 3d screening formula used."]
    if z > 30:
        notes.append(f"Z = {z} is not a 3d metal: the 3d screening formula gives an unphysical Z* "
                     f"for this element.")
    return ZStar(z - (13.5894 + 0.2693 * n_dm1), "generic 3d formula", tuple(notes))
