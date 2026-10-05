"""Save EPR spectra to Bruker BES3T (.DTA/.DSC) format.

Port of EasySpin's ``eprsave.m`` — writes data that can be loaded back with
:func:`torchspin.eprload` or opened in Bruker's Xepr / WinEPR software.

Format overview (Bruker BES3T)
-------------------------------
* `.DSC` — ASCII descriptor file with key=value parameters.
* `.DTA` — Binary data file, big-endian IEEE 754 doubles by default.

The minimal descriptor written here covers all parameters that
:func:`torchspin.eprload` reads, plus the mandatory version header.
"""
from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike


def eprsave(
    filename: Union[str, Path],
    x: ArrayLike,
    y: ArrayLike,
    params: Optional[dict[str, Any]] = None,
    title: str = "",
    xunit: str = "mT",
) -> None:
    """Write an EPR spectrum to Bruker BES3T (.DTA / .DSC) format.

    Parameters
    ----------
    filename:
        Output file path.  The extension is replaced with ``.DTA`` and
        ``.DSC`` automatically — do not include the extension.
    x:
        Abscissa axis (field in mT, or time/frequency in other units).
        Must be uniformly spaced.
    y:
        Spectrum intensity, same length as *x*.  Complex data is written
        as interleaved real/imaginary pairs (IKKF=CPLX).
    params:
        Optional dict of additional parameters to write into the ``.DSC``
        file.  Keys and values are written as ``KEY  value`` lines.
        Values already computed from *x* and *y* (XMIN, XWID, XPTS, …)
        take precedence over anything in *params*.
    title:
        Optional experiment title (written as TITL).
    xunit:
        X-axis unit string written to XUNI.  Default ``'mT'``.

    Examples
    --------
    >>> B, spec = pepper(sys, exp)
    >>> eprsave('output/my_sim', B.numpy(), spec.numpy(), title='S=1/2 sim')
    >>> x, y, p = eprload('output/my_sim.DTA')  # round-trip

    Notes
    -----
    Data is written as big-endian IEEE 754 double precision (BSEQ=BIG,
    IRFMT=D), which maximizes compatibility with EasySpin and Xepr.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    is_complex = np.iscomplexobj(y)
    y = np.asarray(y, dtype=(np.complex128 if is_complex else np.float64)).ravel()

    if len(x) != len(y):
        raise ValueError(
            f"eprsave: x ({len(x)}) and y ({len(y)}) must have the same length."
        )
    if len(x) < 2:
        raise ValueError("eprsave: need at least 2 points.")

    # ── Derive axis parameters ────────────────────────────────────────────
    npts = len(x)
    xmin = float(x[0])
    xwid = float(x[-1]) - xmin   # sweep width

    base = Path(filename).with_suffix("")  # strip any extension
    dta_path = base.with_suffix(".DTA")
    dsc_path = base.with_suffix(".DSC")
    base.parent.mkdir(parents=True, exist_ok=True)

    # ── Write binary .DTA ─────────────────────────────────────────────────
    with open(dta_path, "wb") as f:
        if is_complex:
            interleaved = np.empty(2 * npts, dtype=np.float64)
            interleaved[0::2] = y.real
            interleaved[1::2] = y.imag
            f.write(struct.pack(f">{2 * npts}d", *interleaved))
        else:
            f.write(struct.pack(f">{npts}d", *y.real))

    # ── Build .DSC parameter dict ─────────────────────────────────────────
    now = datetime.now()
    dsc: dict[str, Any] = {}

    # Merge user-supplied params first (our computed values will overwrite conflicts)
    if params is not None:
        dsc.update(params)

    # Mandatory / computed entries (always written, override user params)
    dsc.update({
        "BSEQ": "BIG",
        "IKKF": "CPLX" if is_complex else "REAL",
        "IRFMT": "D",
        "IIFMT": "D" if is_complex else None,
        "XTYP": "IDX",
        "YTYP": "NODATA",
        "ZTYP": "NODATA",
        "XPTS": npts,
        "XMIN": xmin,
        "XWID": xwid,
        "XUNI": f"'{xunit}'",
        "TITL": f"'{title}'" if title else "''",
        "CDATE": now.strftime("%Y/%m/%d"),
        "CTIME": now.strftime("%H:%M:%S"),
    })

    # ── Write .DSC ────────────────────────────────────────────────────────
    with open(dsc_path, "w", encoding="latin-1") as f:
        # Required header block
        f.write("#DESC\t1.0 * DESCRIPTOR INFORMATION ***********************\n")
        f.write("*\n")
        f.write("*\tDSC is the first two bytes of the file. Do not delete them.\n")
        f.write("*\n")
        f.write("DSRC\tMAN\n")

        for key, val in dsc.items():
            if val is None:
                continue
            if isinstance(val, float):
                # Write floats with enough precision
                f.write(f"{key}\t{val:.10g}\n")
            else:
                f.write(f"{key}\t{val}\n")

        f.write("*\n")
        f.write("*\tEnd of descriptor\n")
        f.write("#XYDATA\n")


def eprsave_info(filename: Union[str, Path]) -> None:
    """Print a summary of what eprsave wrote (for debugging).

    Parameters
    ----------
    filename:
        Base path (with or without extension).
    """
    base = Path(filename).with_suffix("")
    dsc_path = base.with_suffix(".DSC")
    dta_path = base.with_suffix(".DTA")

    print(f"DTA: {dta_path}  ({dta_path.stat().st_size} bytes)")
    print(f"DSC: {dsc_path}")
    with open(dsc_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip()
            if line and not line.startswith("*"):
                print(f"  {line}")
