"""Load a Bruker CW spectrum, read the acquisition parameters, plot it, and fit it.

Uses the BES3T file ``tests/eprfiles/strong1.dsc/.dta`` (Bruker "strong pitch" standard: one
line, X-band).  ``eprload`` returns the field axis (in the file's units,
gauss here), the spectrum and the parameter dictionary; the microwave frequency, field range,
modulation amplitude, power and gain come from that dictionary — no manual entry.

Steps: 1. load + parameters  2. baseline  3. ``garlic`` fit of g and the Gaussian/Lorentzian
widths with ``esfit``  4. fit report and overlay.  Swap ``FILE`` for your own .DSC/.DTA or
.par/.spc pair; ``load()`` reads the acquisition parameters from either format.

EasySpin equivalent: eprload + garlic + esfit.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem, Experiment, garlic, esfit, eprload
from torchspin.esfit import FitOptions

REPO = Path(__file__).resolve().parents[2]
FILE = REPO / 'tests' / 'eprfiles' / 'strong1.dsc'     # .dsc or .dta: either name works


def load(path=FILE):
    """Return field (mT), spectrum, mwFreq (GHz) and the raw parameter dict."""
    x, y, prm = eprload(path)
    y = np.asarray(np.real(y), dtype=float)
    # WinEPR (.par) keys: MF = microwave frequency (GHz), HCF/HSW = centre field / sweep (G),
    # RMA = modulation amplitude (G), MP = power (mW), RRG = receiver gain, JSD = scans.
    # BES3T (.DSC) keys: MWFQ (Hz), XMIN/XWID (G), B0MA (T), MWPW (W), RCAG (dB), AVGS.
    if 'MWFQ' in prm:
        mwFreq = float(prm['MWFQ']) / 1e9
        info = dict(power_mW=float(prm.get('MWPW', np.nan)) * 1e3, mod_amp_mT=float(prm.get('B0MA', np.nan)) * 1e3,
                    gain=prm.get('RCAG'), scans=prm.get('AVGS'), temperature_K=prm.get('STMP'))
    else:
        mwFreq = float(prm['MF'])
        info = dict(power_mW=prm.get('MP'), mod_amp_mT=float(prm.get('RMA', np.nan)) / 10.0,
                    gain=prm.get('RRG'), scans=prm.get('JSD'), temperature_K=prm.get('TE'))
    unit = prm.get('XUNI', 'G')
    B_mT = np.asarray(x, dtype=float) / (10.0 if unit.upper().startswith('G') else 1.0)
    return B_mT, y, mwFreq, info, prm


def baseline(B, y, edge=0.1):
    """Subtract a linear baseline fitted to the outer ``edge`` fraction of the sweep."""
    n = len(B); k = max(4, int(edge * n))
    idx = np.r_[:k, n - k:n]
    c = np.polyfit(B[idx], y[idx], 1)
    return y - np.polyval(c, B)


def run():
    B, y_raw, mwFreq, info, prm = load()
    y = baseline(B, y_raw)
    exp = Experiment(mwFreq=mwFreq, Range=[float(B[0]), float(B[-1])], nPoints=len(B), Harmonic=1)
    print(f'loaded {FILE.name}: {len(B)} points, {B[0]:.1f}–{B[-1]:.1f} mT, mwFreq {mwFreq:.4f} GHz, {info}')

    # starting g from the zero crossing of the derivative line (g = h ν / μB B)
    i0 = int(np.argmax(y)); i1 = i0 + int(np.argmin(y[i0:]))
    B_cross = B[i0] - y[i0] * (B[i1] - B[i0]) / (y[i1] - y[i0])
    g_est = 71.44773 * mwFreq / B_cross                 # h/μB = 71.44773 mT/GHz
    print(f'zero crossing at {B_cross:.3f} mT → g ≈ {g_est:.5f}')

    # one-line model: isotropic g, Gaussian and Lorentzian FWHM (mT)
    def model(p):
        g_iso, lw_g, lw_l = p
        _, spc = garlic(SpinSystem(S=[0.5], g=[[g_iso] * 3], lw=[abs(lw_g), abs(lw_l)]), exp)
        return np.asarray(spc, dtype=float)

    p0 = np.array([g_est, 0.5, 0.5])                    # g_iso, lw Gaussian (mT), lw Lorentzian (mT)
    vary = np.array([0.003, 0.5, 0.5])
    opts = FitOptions(method='global', swarm_size=16, max_iter=20, seed=0, verbosity=0, compute_uncertainties=False)
    res = esfit(y, model, p0, vary=vary, options=opts)
    names = ['g_iso', 'lw Gaussian (mT)', 'lw Lorentzian (mT)']
    print('fit:', {n: round(float(v), 4) for n, v in zip(names, res.pfit)}, f'rmsd {res.rmsd:.4g}')
    return B, y, res, mwFreq


def _plot(B, y, res):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax1.plot(B, y, 'k', lw=0.9, label='experiment (baseline-corrected)')
    ax1.plot(B, res.fit, 'r', lw=1.2, label='garlic fit')
    ax1.legend(); ax1.set_ylabel('intensity (a.u.)'); ax1.set_title('Bruker strong pitch, X-band CW (BES3T file)')
    ax2.plot(B, y - res.fit, 'b', lw=0.8); ax2.axhline(0, color='k', lw=0.5); ax2.set_xlabel('B (mT)'); ax2.set_ylabel('residual')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    B, y, res, mwFreq = run()
    out = Path(__file__).with_suffix('.png')
    _plot(B, y, res).savefig(out, dpi=110)
    print('saved', out)
