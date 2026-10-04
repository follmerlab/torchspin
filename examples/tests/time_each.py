"""Time each torchspin_examples run() one at a time, printing as each finishes.

Usage:
    python torchspin_examples/tests/time_each.py
    python torchspin_examples/tests/time_each.py --category liquids
    python torchspin_examples/tests/time_each.py --timeout 120   # skip examples over 120s
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(EXAMPLES_ROOT.parent))

ALL_EXAMPLES = [
    # solidstate
    "solidstate/broaden",
    "solidstate/gstrain",
    "solidstate/triplet_naphthalene",
    "solidstate/triplet_c60",
    "solidstate/freqsweep",
    "solidstate/triplet_triphenylbenzene",
    "solidstate/temperature",
    "solidstate/iron_highspin",
    "solidstate/copper_nitrogens",
    "solidstate/eespins",
    "solidstate/chromium_iii",
    "solidstate/cuedta",
    "solidstate/copper_sxq",
    "solidstate/matrixperturb",
    "solidstate/inhomogeneous",
    "solidstate/triplet_halffieldintensity",
    "solidstate/gd_transitions",
    "solidstate/trifluormethyl",
    "solidstate/mn_strain",
    # liquids
    "liquids/biphenyl",
    "liquids/fremysalt",
    "liquids/fast_fnb",
    "liquids/naphthalene",
    "liquids/isotopemix",
    "liquids/biarylradical",
    "liquids/cobalttrimer",
    "liquids/nitroxide_ftcorr",
    # magnetometry
    "magnetometry/curie_law",
    "magnetometry/squid_basic",
    "magnetometry/magnetization_tempdep",
    "magnetometry/susceptibility",
    "magnetometry/effectivemagneticmoment",
    # slowmotion
    "slowmotion/nitroxide_basic",
    "slowmotion/nitroxide_tcorr",
    "slowmotion/nitroxide_frq",
    "slowmotion/tempone",
    "slowmotion/tumbling",
    "slowmotion/slow_fast",
    # endor
    "endor/endorsimple",
    "endor/endorseparate",
    "endor/excitew",
    "endor/manynuclei",
    "endor/endorperturb",
    # fitting
    "fitting/basicfit",
    "fitting/multicomponents",
    "fitting/twocompfit",
    "fitting/pentacenefit",
    "fitting/globallocal",
    "fitting/fit_multifreq",
    # analysis
    "analysis/denoise",
]

MATLAB_TIMES = {
    "solidstate/broaden": 0.18,
    "solidstate/gstrain": 0.55,
    "solidstate/triplet_naphthalene": 0.12,
    "solidstate/triplet_c60": 0.14,
    "solidstate/freqsweep": 0.29,
    "solidstate/triplet_triphenylbenzene": 0.13,
    "solidstate/temperature": 0.55,
    "solidstate/iron_highspin": 5.17,
    "solidstate/copper_nitrogens": 1.83,
    "solidstate/eespins": 0.21,
    "solidstate/chromium_iii": 8.40,
    "solidstate/cuedta": 0.35,
    "solidstate/copper_sxq": 0.52,
    "solidstate/matrixperturb": 6.31,
    "solidstate/inhomogeneous": 1.20,
    "solidstate/triplet_halffieldintensity": 3.10,
    "solidstate/gd_transitions": 0.27,
    "solidstate/trifluormethyl": 0.38,
    "solidstate/mn_strain": 4.20,
    "liquids/biphenyl": 0.04,
    "liquids/fremysalt": 0.03,
    "liquids/fast_fnb": 0.15,
    "liquids/naphthalene": 0.08,
    "liquids/isotopemix": 0.04,
    "liquids/biarylradical": 0.12,
    "liquids/cobalttrimer": 0.06,
    "liquids/nitroxide_ftcorr": 0.12,
    "magnetometry/curie_law": 0.06,
    "magnetometry/squid_basic": 0.18,
    "magnetometry/magnetization_tempdep": 0.22,
    "magnetometry/susceptibility": 0.25,
    "magnetometry/effectivemagneticmoment": 0.14,
    "slowmotion/nitroxide_basic": 0.27,
    "slowmotion/nitroxide_tcorr": 0.80,
    "slowmotion/nitroxide_frq": 1.10,
    "slowmotion/tempone": 0.24,
    "slowmotion/tumbling": 1.40,
    "slowmotion/slow_fast": 0.35,
    "endor/endorsimple": 0.09,
    "endor/endorseparate": 0.11,
    "endor/excitew": 0.33,
    "endor/manynuclei": 0.14,
    "endor/endorperturb": 0.20,
    "fitting/basicfit": 8.50,
    "fitting/multicomponents": 12.00,
    "fitting/twocompfit": 10.00,
    "fitting/pentacenefit": 6.00,
    "fitting/globallocal": 18.00,
    "fitting/fit_multifreq": 20.00,
    "analysis/denoise": 0.02,
}


def _import(subdir, name):
    path = EXAMPLES_ROOT / subdir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_example_worker(q, subdir, name):
    try:
        mod = _import(subdir, name)
        t0 = time.perf_counter()
        mod.run()
        q.put(("ok", time.perf_counter() - t0))
    except Exception as e:
        q.put(("error", str(e)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", nargs="+",
                        choices=["solidstate", "liquids", "magnetometry",
                                 "slowmotion", "endor", "fitting", "analysis"])
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Skip examples that run longer than this many seconds")
    args = parser.parse_args()

    examples = ALL_EXAMPLES
    if args.category:
        examples = [e for e in examples if e.split("/")[0] in args.category]

    print(f"\n{'='*70}")
    print("torchspin per-example timing  (Python vs MATLAB EasySpin)")
    print(f"{'='*70}\n")
    print(f"{'Example':<42}  {'Python (s)':>10}  {'MATLAB (s)':>10}  {'Ratio':>8}")
    print("-" * 78)

    results = []
    for key in examples:
        subdir, name = key.split("/")
        matlab_t = MATLAB_TIMES.get(key)
        sys.stdout.write(f"{key:<42}  {'...':<10}  ")
        sys.stdout.flush()

        import multiprocessing

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=_run_example_worker, args=(q, subdir, name))
        t0 = time.perf_counter()
        p.start()
        p.join(timeout=args.timeout)
        elapsed = time.perf_counter() - t0

        if p.is_alive():
            p.terminate()
            p.join()
            print(f"\r{key:<42}  {elapsed:10.1f}+s  {matlab_t or 'N/A':>10}  TIMEOUT")
            results.append((key, elapsed, matlab_t, "timeout"))
            continue

        try:
            status, val = q.get_nowait()
        except Exception:
            status, val = "error", "no result"

        if status == "error":
            print(f"\r{key:<42}  {'ERROR':>10}  {'':>10}  {str(val)[:30]}")
            results.append((key, None, matlab_t, "error"))
        else:
            elapsed = val
            if matlab_t:
                ratio = elapsed / matlab_t
                flag = " !" if ratio > 10 else (" ~" if ratio > 3 else "  ")
                print(f"\r{key:<42}  {elapsed:10.3f}  {matlab_t:10.3f}  {ratio:7.2f}x{flag}")
            else:
                print(f"\r{key:<42}  {elapsed:10.3f}  {'N/A':>10}  {'N/A':>8}")
            results.append((key, elapsed, matlab_t, "ok"))

    print("-" * 78)
    valid = [(k, py, ml) for k, py, ml, st in results if st == "ok" and ml is not None]
    if valid:
        ratios = sorted([py / ml for _, py, ml in valid])
        print(f"\nMedian ratio: {ratios[len(ratios)//2]:.2f}x")
        print(f"Mean ratio:   {sum(ratios)/len(ratios):.2f}x")
        slow = [(k, py/ml) for k, py, ml in valid if py/ml > 10]
        if slow:
            print(f"\nExamples >10x slower than MATLAB (optimization targets):")
            for k, r in sorted(slow, key=lambda x: -x[1]):
                print(f"  {k:<42}  {r:.1f}x")


if __name__ == "__main__":
    main()
