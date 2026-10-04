"""Timing benchmark for torchspin_examples vs MATLAB EasySpin reference times.

Usage:
    python torchspin_examples/tests/benchmark_timing.py
    python torchspin_examples/tests/benchmark_timing.py --category solidstate
    python torchspin_examples/tests/benchmark_timing.py --all   # include slow examples

Reports wall-clock time for each example's run() call and the MATLAB speedup ratio.

MATLAB reference times were measured on a MacBook Pro M2 (2023) running
EasySpin 6.0.0 in MATLAB R2024a.  Times are the mean of 3 runs.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(EXAMPLES_ROOT.parent))


# ---------------------------------------------------------------------------
# MATLAB reference wall-clock times (seconds, measured 2026-03-16)
# Source: easyspin_example_timings_2026-03-16.csv (sequential MATLAB run)
# ---------------------------------------------------------------------------
MATLAB_TIMES = {
    # solidstate — from MATLAB measurements
    "solidstate/broaden":                   1.780,
    "solidstate/gstrain":                   2.997,
    "solidstate/iron_highspin":             1.225,
    "solidstate/triplet_naphthalene":       0.487,
    "solidstate/triplet_c60":               0.263,  # triplet_C60fullerene.m
    "solidstate/temperature":               0.687,  # temp.m
    "solidstate/copper_nitrogens":          0.478,
    "solidstate/eespins":                   0.547,
    "solidstate/freqsweep":                 0.189,
    "solidstate/chromium_iii":             15.762,
    "solidstate/cuedta":                    0.732,
    "solidstate/copper_sxq":               1.250,
    "solidstate/matrixperturb":             1.498,
    "solidstate/inhomogeneous":            17.398,
    "solidstate/triplet_halffieldintensity":9.468,
    "solidstate/triplet_triphenylbenzene":  0.264,
    "solidstate/gd_transitions":            0.421,
    "solidstate/trifluormethyl":            0.177,
    "solidstate/mn_strain":                 0.564,
    # liquids — from MATLAB measurements
    "liquids/biphenyl":                     0.100,
    "liquids/fremysalt":                    0.149,
    "liquids/fast_fnb":                     0.158,
    "liquids/naphthalene":                  0.145,
    "liquids/isotopemix":                   0.250,
    "liquids/biarylradical":                0.121,
    "liquids/cobalttrimer":                 0.213,
    "liquids/nitroxide_ftcorr":             0.159,
    # magnetometry — from MATLAB measurements
    "magnetometry/curie_law":               0.149,
    "magnetometry/squid_basic":             0.291,
    "magnetometry/magnetization_tempdep":   0.173,
    "magnetometry/susceptibility":          0.150,
    "magnetometry/effectivemagneticmoment": 0.130,
    # slowmotion — from MATLAB measurements
    "slowmotion/nitroxide_basic":           0.252,
    "slowmotion/nitroxide_tcorr":           6.582,
    "slowmotion/nitroxide_frq":             0.506,
    "slowmotion/tempone":                   0.209,
    "slowmotion/tumbling":                  0.337,
    "slowmotion/slow_fast":                 0.267,
    # endor — from MATLAB measurements
    "endor/endorsimple":                    0.176,
    "endor/endorseparate":                  0.288,
    "endor/excitew":                        0.427,
    "endor/manynuclei":                     0.396,
    "endor/endorperturb":                   2.483,
    # fitting — from MATLAB measurements
    "fitting/basicfit":                    12.195,
    "fitting/multicomponents":              0.899,
    "fitting/twocompfit":                  11.479,
    "fitting/pentacenefit":                 0.815,
    "fitting/globallocal":                  1.452,
    "fitting/fit_multifreq":               49.279,
    # analysis — from MATLAB measurements
    "analysis/denoise":                     0.162,
}

# Examples that are computationally heavy (skipped unless --all)
SLOW_EXAMPLES = {
    "solidstate/chromium_iii",
    "solidstate/iron_highspin",
    "solidstate/matrixperturb",
    "solidstate/gd_transitions",
    "solidstate/trifluormethyl",
    "solidstate/mn_strain",
    "solidstate/triplet_halffieldintensity",
    "fitting/basicfit",
    "fitting/multicomponents",
    "fitting/twocompfit",
    "fitting/pentacenefit",
    "fitting/globallocal",
    "fitting/fit_multifreq",
}


def _import(subdir, name):
    path = EXAMPLES_ROOT / subdir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_benchmark(categories=None, include_slow=False):
    all_keys = sorted(MATLAB_TIMES.keys())
    if categories:
        all_keys = [k for k in all_keys if k.split("/")[0] in categories]
    if not include_slow:
        all_keys = [k for k in all_keys if k not in SLOW_EXAMPLES]

    results = []
    header = f"{'Example':<42}  {'Python (s)':>10}  {'MATLAB (s)':>10}  {'Ratio':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for key in all_keys:
        subdir, name = key.split("/")
        matlab_t = MATLAB_TIMES.get(key)

        try:
            mod = _import(subdir, name)
            t0 = time.perf_counter()
            mod.run()
            elapsed = time.perf_counter() - t0

            if matlab_t is not None:
                ratio = elapsed / matlab_t
                ratio_str = f"{ratio:8.2f}x"
                flag = " !" if ratio > 10 else ("  " if ratio < 3 else " ~")
            else:
                ratio_str = "       N/A"
                flag = "  "

            matlab_str = f"{matlab_t:10.3f}" if matlab_t else "       N/A"
            print(f"{key:<42}  {elapsed:10.3f}  {matlab_str}  {ratio_str}{flag}")
            results.append((key, elapsed, matlab_t))

        except Exception as e:
            print(f"{key:<42}  {'ERROR':>10}  {'':>10}  {'':>8}  ({e!s:.60s})")

    print(sep)

    # Summary stats
    valid = [(k, py, ml) for k, py, ml in results if ml is not None]
    if valid:
        ratios = [py / ml for _, py, ml in valid]
        print(f"\nMedian speedup ratio (Python/MATLAB): {sorted(ratios)[len(ratios)//2]:.2f}x")
        print(f"Mean   speedup ratio (Python/MATLAB): {sum(ratios)/len(ratios):.2f}x")
        print(f"Best:  {min(ratios):.2f}x  ({valid[ratios.index(min(ratios))][0]})")
        print(f"Worst: {max(ratios):.2f}x  ({valid[ratios.index(max(ratios))][0]})")
        print()
        print("Legend:  ratio < 3x = acceptable,  ~ = 3-10x,  ! = >10x (optimization target)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark torchspin_examples vs MATLAB")
    parser.add_argument("--category", nargs="+",
                        choices=["solidstate", "liquids", "magnetometry",
                                 "slowmotion", "endor", "fitting", "analysis"],
                        help="Run only specified categories")
    parser.add_argument("--all", action="store_true",
                        help="Include slow examples (fitting, high-spin powder)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("torchspin_examples timing benchmark  (Python vs MATLAB EasySpin)")
    print(f"{'='*70}\n")
    run_benchmark(categories=args.category, include_slow=args.all)


if __name__ == "__main__":
    main()
