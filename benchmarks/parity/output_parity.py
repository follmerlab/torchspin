"""Output parity harness for performance work (docs/v1_acceleration/ROADMAP.md).

Runs a fixed set of simulations that cover the code paths touched by performance
changes and saves every numeric output, so a change can be checked against the
outputs of the code before it:

    # before the change (any checkout, e.g. main)
    python benchmarks/parity/output_parity.py save base_1t.npz --threads 1 --repo /path/to/main
    # after the change
    python benchmarks/parity/output_parity.py compare base_1t.npz --threads 1 --mode exact

``--mode exact`` requires every array to be identical (``np.array_equal``, NaN
patterns included); ``--mode roundoff`` allows ``max|a - b| <= tol * max|a|`` per
case (``--tol``, default 1e-10).  The thread count is applied to torch and to every
native pool before torch is imported, so the same file can be recorded and compared
at 1 and at 8 threads.  ``--only`` selects cases by substring.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['save', 'compare', 'list'])
    ap.add_argument('path', nargs='?')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--repo', default=None, help='torchspin checkout to import (default: this one)')
    ap.add_argument('--mode', choices=['exact', 'roundoff'], default='exact')
    ap.add_argument('--tol', type=float, default=1e-10)
    ap.add_argument('--only', default='', help='comma-separated substrings of case names')
    return ap.parse_args()


def _cases():
    """name -> zero-argument callable returning the outputs to compare."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    import torch

    from torchspin import (SpinSystem, Experiment, Options, pepper, garlic, chili, saffron, salt, curry,
                           cardamom, CardamomPar, CardamomOptions, esfit, FitOptions)
    from torchspin.chili import ChiliOptions
    from torchspin.pepper_autograd import pepper_autograd
    from torchspin.saffron import PulseExperiment, SaffronOptions

    repo = Path(sys.modules['torchspin'].__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('_mk', repo / 'tests' / 'data' / 'generate_missing_knots_refs.py')
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)

    O = lambda **kw: Options(Verbosity=0, **kw)          # noqa: E731
    nitro = dict(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]], lwpp=[0.3, 0])
    exp_n = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024)
    cu2n = dict(S=[0.5], g=[2.05, 2.05, 2.25], Nucs='63Cu,14N,14N',
                A=[[30, 30, 500], [40, 40, 40], [40, 40, 40]], lwpp=[0.5, 0])
    exp_cu = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=1024)
    C = {}

    # ---------------------------------------------------------------- pepper
    C['pepper_nitroxide_31x4'] = lambda: pepper(SpinSystem(**nitro), exp_n, O(GridSize=[31, 4]))
    C['pepper_nitroxide_Ci50'] = lambda: pepper(
        SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]], Nucs='14N', A=[[10.0, 10.0, 95.0]], lw=[1.0, 0.0]),
        Experiment(mwFreq=9.5, Range=[330.0, 350.0], nPoints=1024, Harmonic=1), O(GridSize=50, GridSymmetry='Ci'))
    C['pepper_cu_rhombic_default'] = lambda: pepper(
        SpinSystem(S=[0.5], g=[2.05, 2.06, 2.25], Nucs='63Cu', A=[[30, 40, 500]], lwpp=[0.8, 0]),
        Experiment(mwFreq=9.5, Range=[250, 360], nPoints=1024), O())
    C['pepper_cu2n_10x4'] = lambda: pepper(SpinSystem(**cu2n), exp_cu, O(GridSize=[10, 4]))
    C['pepper_cu2n_10x4_separate'] = lambda: pepper(SpinSystem(**cu2n), exp_cu, O(GridSize=[10, 4], separate='transitions'))
    C['pepper_cu2n_hybrid'] = lambda: pepper(SpinSystem(**cu2n), exp_cu, O(GridSize=[10, 4], Method='hybrid',
                                                                            HybridCoreNuclei=[1]))
    C['pepper_mn_S52_10x4'] = lambda: pepper(
        SpinSystem(S=[2.5], g=2.0, D=[[600, 100]], Nucs='55Mn', A=[[250, 250, 250]], lwpp=[1, 0]),
        Experiment(mwFreq=9.5, Range=[260, 360], nPoints=1024), O(GridSize=[10, 4]))
    C['pepper_perturb_45x1'] = lambda: pepper(
        SpinSystem(S=[0.5], g=[2.0023, 2.0025, 2.0035], Nucs='14N,1H,1H',
                   A=[[20, 20, 90], [8, 8, 12], [5, 5, 7]], lwpp=[0.2, 0]),
        Experiment(mwFreq=9.5, Range=[330, 350], nPoints=2048), O(GridSize=[45, 1], Method='perturb2'))
    C['pepper_strain_21x4'] = lambda: pepper(
        SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]],
                   gStrain=[[0.003, 0.002, 0.001]], HStrain=[10, 10, 30], lwpp=[0.3, 0]),
        Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024), O(GridSize=[21, 4]))
    C['pepper_gstrain_radical'] = lambda: pepper(
        SpinSystem(S=[0.5], g=[[2.0104, 2.0074, 2.0026]], gStrain=[[0.001, 0.0008, 0.0005]], lw=[0.0, 0.0]),
        Experiment(mwFreq=9.5, Range=[332.0, 342.0], nPoints=1024, Harmonic=1), O(GridSize=31))
    C['pepper_triplet_zfs'] = lambda: pepper(
        SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[200.0, 200.0, -400.0], lw=[1.0, 0.0]),
        Experiment(mwFreq=9.5, Range=[300.0, 380.0], nPoints=1024, Harmonic=0), O(GridSize=50))
    C['pepper_freqsweep_cu'] = lambda: pepper(
        SpinSystem(S=[0.5], g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[[30, 30, 500]], lwpp=[0.5, 0]),
        Experiment(Field=320.0, mwRange=[8.8, 10.4], nPoints=2048), O(GridSize=[19, 4]))
    for nm in ('case1', 'case2', 'case3'):
        def _mk(nm=nm, func=pepper):
            s, e, okw = getattr(mk, nm)()
            return func(s, e, O(GridSize=[19, 4], **okw))
        C[f'pepper_missing_knots_{nm}'] = _mk
        C[f'pepper_autograd_missing_knots_{nm}'] = lambda nm=nm: _mk(nm, pepper_autograd)

    # ---------------------------------------------------------------- gradients
    def _grad_nitroxide():
        g = torch.tensor([2.008, 2.006, 2.003], dtype=torch.float64, requires_grad=True)
        A = torch.tensor([[20.0, 20.0, 85.0]], dtype=torch.float64, requires_grad=True)
        lw = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        _, y = pepper_autograd(SpinSystem(S=[0.5], g=g, Nucs='14N', A=A, lwpp=[lw, 0.0]), exp_n, O(GridSize=[31, 4]))
        (y ** 2).sum().backward()
        return y.detach(), g.grad, A.grad, lw.grad
    C['grad_pepper_autograd_nitroxide'] = _grad_nitroxide

    def _grad_strain():
        H = torch.tensor([10.0, 10.0, 30.0], dtype=torch.float64, requires_grad=True)
        _, y = pepper_autograd(SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]],
                                          HStrain=H, lwpp=[0.3, 0]), exp_n, O(GridSize=[15, 4]))
        (y ** 2).sum().backward()
        return y.detach(), H.grad
    C['grad_pepper_autograd_hstrain'] = _grad_strain

    # ---------------------------------------------------------------- fitting
    _fit_exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512)

    def _model(p):
        s = SpinSystem(S=[0.5], g=[float(p[0]), float(p[1]), float(p[2])], Nucs='14N', A=[[20, 20, 85]],
                       lwpp=[float(p[3]), 0])
        return pepper(s, _fit_exp, O(GridSize=[15, 4]))[1].numpy()

    def _fit(method, **kw):
        data = _model(np.array([2.0080, 2.0060, 2.0030, 0.30]))
        r = esfit(data, _model, p0=np.array([2.0070, 2.0068, 2.0024, 0.40]),
                  vary=np.array([0.003, 0.003, 0.003, 0.2]),
                  options=FitOptions(method=method, verbosity=0, **kw))
        return r
    C['esfit_levmar_4p'] = lambda: _fit('levmar', max_iter=4, compute_uncertainties=True)
    C['esfit_simplex_4p'] = lambda: _fit('simplex', max_iter=25)

    # ---------------------------------------------------------------- other simulators
    C['garlic_nitroxide_fast'] = lambda: garlic(
        SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]], Nucs='14N', A=[[16.0, 16.0, 95.0]], tcorr=1e-10, lw=[0.1, 0.0]),
        Experiment(mwFreq=9.5, Range=[336.0, 342.0], nPoints=1024, Harmonic=1), O())
    C['chili_nitroxide_1ns'] = lambda: chili(
        SpinSystem(S=[0.5], g=[[2.009, 2.006, 2.002]], Nucs='14N', A=[[16.0, 16.0, 95.0]], tcorr=1e-9, lw=[0.0, 0.0]),
        Experiment(mwFreq=9.5, Range=[332.0, 342.0], nPoints=256, Harmonic=0), ChiliOptions(LLMK=[14, 7, 2, 6]))
    C['chili_nitroxide_2nuc'] = lambda: chili(
        SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N,1H', A=[[20, 20, 85], [5, 5, 8]], tcorr=3e-8,
                   lw=[0.1, 0.1]), Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512))
    C['chili_potential'] = lambda: chili(
        SpinSystem(S=[0.5], g=[2.008, 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]], tcorr=2e-8, lw=[0.1, 0.1],
                   Potential=[[2, 0, 0, 1.5]]), Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512))
    C['cardamom_diffusion_seeded'] = lambda: cardamom(
        SpinSystem(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20, 20, 85]], tcorr=1e-9),
        Experiment(mwFreq=9.5, Range=[332, 352], nPoints=256, Harmonic=0),
        CardamomPar(Model='diffusion', nTraj=20, nSteps=300, dtSpin=1e-10, dtSpatial=1e-10, seed=7),
        CardamomOptions(Method='fast', Verbosity=0))
    C['saffron_2pESEEM'] = lambda: saffron(
        SpinSystem(S=[0.5], g=[[2.0023] * 3], Nucs='1H', A=[[3.0, 3.0, 9.0]]),
        PulseExperiment(Sequence='2pESEEM', Field=350.0, dt=0.010, tau=0.0, nPoints=256),
        SaffronOptions(GridSize=30, TimeDomain=True))
    C['saffron_3pESEEM'] = lambda: saffron(
        SpinSystem(S=[0.5], g=[[2.0023] * 3], Nucs='1H', A=[[3.0, 3.0, 9.0]]),
        PulseExperiment(Sequence='3pESEEM', Field=350.0, dt=0.010, tau=0.1, nPoints=256),
        SaffronOptions(GridSize=30))
    C['saffron_hyscore_td_48'] = lambda: saffron(
        SpinSystem(S=[0.5], g=[[2.0023] * 3], Nucs='1H', A=[[3.0, 3.0, 9.0]]),
        PulseExperiment(Sequence='HYSCORE', Field=324.9, dt=0.050, nPoints=[48, 48], tau=0.08, t1=0.1, t2=0.1),
        SaffronOptions(GridSize=12, TimeDomain=True))
    C['saffron_hyscore_14N_1H_128'] = lambda: saffron(
        SpinSystem(S=[0.5], g=[2.0023] * 3, Nucs='14N,1H', A=[[3, 3, 6], [2, 2, 8]], Q=[[-0.5, -0.5, 1.0], [0, 0, 0]]),
        PulseExperiment(Sequence='HYSCORE', Field=350, dt=0.016, nPoints=128, tau=0.1), SaffronOptions(GridSize=31))
    C['salt_14N_wband'] = lambda: salt(
        SpinSystem(S=[0.5], g=[[2.2, 2.2, 2.0]], Nucs='14N', A=[[4.0, 4.0, 5.0]], Q=[[0.84, 0.84, -1.68]]),
        Experiment(Field=3394.0, mwFreq=95.0), O(GridSize=19), freq_range=(5, 20), n_points=512, lw_mhz=0.2)
    C['curry_S52'] = lambda: curry(SpinSystem(S=[2.5], g=[2.0], D=[[3000.0, 300.0]]), [100.0],
                                   np.linspace(2, 300, 40).tolist())
    return C


# attributes that are not results (wall-clock timings)
_VOLATILE = {'elapsed_s'}


def _flatten(obj, out, prefix=''):
    """Collect every numeric array/scalar in ``obj`` under stable keys."""
    import numpy as np
    import torch
    if obj is None or isinstance(obj, (str, bytes)):
        return
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj.detach().cpu().numpy().copy()
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in 'biufc':
            out[prefix] = obj.copy()
    elif isinstance(obj, (bool, int, float, complex, np.number)):
        out[prefix] = np.asarray(obj)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(v, out, f'{prefix}[{i}]')
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _flatten(obj[k], out, f'{prefix}.{k}')
    elif hasattr(obj, '__dict__'):
        for k in sorted(vars(obj)):
            if not k.startswith('_') and k not in _VOLATILE:
                _flatten(getattr(obj, k), out, f'{prefix}.{k}')


def main():
    a = _parse()
    for k in _THREAD_ENV:
        os.environ[k] = str(a.threads)
    if a.repo:
        sys.path.insert(0, os.path.abspath(os.path.expanduser(a.repo)))
    else:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import torchspin  # noqa: F401  (before torch: OMP_WAIT_POLICY)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    print(f'torchspin from {torchspin.__file__}, threads {a.threads}', flush=True)
    cases = _cases()
    if a.action == 'list':
        print('\n'.join(cases))
        return 0
    sel = [n for n in cases if not a.only or any(s in n for s in a.only.split(','))]
    results, times = {}, {}
    for name in sel:
        t0 = time.perf_counter()
        out = {}
        _flatten(cases[name](), out, name)
        times[name] = time.perf_counter() - t0
        results[name] = out
        print(f'  {name:40s} {times[name]:7.3f} s  ({len(out)} arrays)', flush=True)
    if a.action == 'save':
        flat = {k: v for out in results.values() for k, v in out.items()}
        np.savez_compressed(a.path, **flat)
        print(f'saved {len(flat)} arrays to {a.path}')
        return 0
    ref = np.load(a.path)
    bad = 0
    for name, out in results.items():
        keys_ref = {k for k in ref.files if (k == name or k.startswith(name + '[') or k.startswith(name + '.'))
                    and k.rsplit('.', 1)[-1] not in _VOLATILE}
        if set(out) != keys_ref:
            print(f'DIFF {name}: output keys differ ({sorted(set(out) ^ keys_ref)[:4]})'); bad += 1
            continue
        worst = 0.0
        for k in out:
            x, y = ref[k], out[k]
            if x.shape != y.shape:
                print(f'DIFF {k}: shape {x.shape} vs {y.shape}'); bad += 1; worst = np.inf; break
            if a.mode == 'exact':
                if not np.array_equal(x, y, equal_nan=x.dtype.kind in 'fc'):
                    d = np.nanmax(np.abs(x.astype(complex) - y.astype(complex))) if x.size else 0.0
                    s = np.nanmax(np.abs(x)) if x.size else 0.0
                    worst = max(worst, d / s if s else d)
            else:
                if not np.array_equal(np.isnan(x), np.isnan(y)) if x.dtype.kind in 'fc' else False:
                    worst = np.inf
                else:
                    d = np.nanmax(np.abs(x.astype(complex) - y.astype(complex))) if x.size else 0.0
                    s = np.nanmax(np.abs(x)) if x.size else 0.0
                    worst = max(worst, d / s if s else d)
        ok = worst == 0.0 if a.mode == 'exact' else worst <= a.tol
        bad += not ok
        print(f'{"ok  " if ok else "DIFF"} {name:40s} max rel diff {worst:.3e}')
    print(f'{len(results) - bad}/{len(results)} cases pass ({a.mode})')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
