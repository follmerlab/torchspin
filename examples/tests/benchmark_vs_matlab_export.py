"""Run matching torchspin examples, time them, and export comparison tables.

Usage:
    python torchspin_examples/tests/benchmark_vs_matlab_export.py
"""
from __future__ import annotations

import csv
import importlib.util
import multiprocessing
import sys
import time
from datetime import date
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXAMPLES_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))


MATLAB_TIMES = {
    "analysis/denoise": 0.162,
    "endor/endorperturb": 2.483,
    "endor/endorseparate": 0.288,
    "endor/endorsimple": 0.176,
    "endor/excitew": 0.427,
    "endor/manynuclei": 0.396,
    "fitting/basicfit": 12.195,
    "fitting/fit_multifreq": 49.279,
    "fitting/globallocal": 1.452,
    "fitting/multicomponents": 0.899,
    "fitting/pentacenefit": 0.815,
    "fitting/twocompfit": 11.479,
    "liquids/biarylradical": 0.121,
    "liquids/biphenyl": 0.100,
    "liquids/cobalttrimer": 0.213,
    "liquids/fast_fnb": 0.158,
    "liquids/fremysalt": 0.149,
    "liquids/isotopemix": 0.250,
    "liquids/naphthalene": 0.145,
    "liquids/nitroxide_ftcorr": 0.159,
    "magnetometry/curie_law": 0.149,
    "magnetometry/effectivemagneticmoment": 0.130,
    "magnetometry/magnetization_tempdep": 0.173,
    "magnetometry/squid_basic": 0.291,
    "magnetometry/susceptibility": 0.150,
    "slowmotion/nitroxide_basic": 0.252,
    "slowmotion/nitroxide_frq": 0.506,
    "slowmotion/nitroxide_tcorr": 6.582,
    "slowmotion/slow_fast": 0.267,
    "slowmotion/tempone": 0.209,
    "slowmotion/tumbling": 0.337,
    "solidstate/broaden": 1.780,
    "solidstate/chromium_iii": 15.762,
    "solidstate/copper_nitrogens": 0.478,
    "solidstate/copper_sxq": 1.250,
    "solidstate/cuedta": 0.732,
    "solidstate/eespins": 0.547,
    "solidstate/freqsweep": 0.189,
    "solidstate/gd_transitions": 0.421,
    "solidstate/gstrain": 2.997,
    "solidstate/inhomogeneous": 17.398,
    "solidstate/iron_highspin": 1.225,
    "solidstate/matrixperturb": 1.498,
    "solidstate/mn_strain": 0.564,
    "solidstate/temperature": 0.687,
    "solidstate/trifluormethyl": 0.177,
    "solidstate/triplet_c60": 0.263,
    "solidstate/triplet_halffieldintensity": 9.468,
    "solidstate/triplet_naphthalene": 0.487,
    "solidstate/triplet_triphenylbenzene": 0.264,
}


def _import_example(subdir: str, name: str):
    path = EXAMPLES_ROOT / subdir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_example(key: str):
    subdir, name = key.split("/")
    module = _import_example(subdir, name)
    t0 = time.perf_counter()
    module.run()
    return time.perf_counter() - t0


def _run_example_worker(queue, key: str):
    try:
        elapsed = _run_example(key)
        queue.put(("pass", elapsed, ""))
    except Exception as exc:
        queue.put(("fail", "", str(exc)))


def _write_csv(rows, path: Path):
    with path.open("w", newline="") as f:
      writer = csv.DictWriter(
          f,
          fieldnames=[
              "example",
              "python_seconds",
              "matlab_seconds",
              "python_over_matlab",
              "status",
              "message",
          ],
      )
      writer.writeheader()
      writer.writerows(rows)


def _write_markdown(rows, path: Path):
    run_date = date.today().isoformat()
    lines = [
        f"# torchspin vs EasySpin Timings ({run_date})",
        "",
        "| Example | Python (s) | MATLAB (s) | Python/MATLAB | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        py = row["python_seconds"]
        ml = row["matlab_seconds"]
        ratio = row["python_over_matlab"]
        py_str = f"{py:.3f}" if isinstance(py, float) else ""
        ml_str = f"{ml:.3f}" if isinstance(ml, float) else ""
        ratio_str = f"{ratio:.3f}x" if isinstance(ratio, float) else ""
        lines.append(
            f"| {row['example']} | {py_str} | {ml_str} | {ratio_str} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    results_dir = REPO_ROOT / "tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_date = date.today().isoformat()

    rows = []
    timeout_seconds = 300.0
    for key in sorted(MATLAB_TIMES):
        print(f"Running {key}...", flush=True)
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_run_example_worker, args=(queue, key))
        proc.start()
        proc.join(timeout=timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            rows.append(
                {
                    "example": key,
                    "python_seconds": "",
                    "matlab_seconds": MATLAB_TIMES[key],
                    "python_over_matlab": "",
                    "status": "timeout",
                    "message": f"Timed out after {timeout_seconds:.0f} s",
                }
            )
            print(f"  TIMEOUT after {timeout_seconds:.0f} s", flush=True)
            continue

        try:
            status, elapsed, message = queue.get_nowait()
        except Exception:
            status, elapsed, message = "fail", "", "No result returned"

        if status == "pass":
            matlab_time = MATLAB_TIMES[key]
            ratio = elapsed / matlab_time
            rows.append(
                {
                    "example": key,
                    "python_seconds": elapsed,
                    "matlab_seconds": matlab_time,
                    "python_over_matlab": ratio,
                    "status": "pass",
                    "message": "",
                }
            )
            print(
                f"  {elapsed:.3f} s  vs MATLAB {matlab_time:.3f} s  ({ratio:.3f}x)",
                flush=True,
            )
        else:
            rows.append(
                {
                    "example": key,
                    "python_seconds": "",
                    "matlab_seconds": MATLAB_TIMES[key],
                    "python_over_matlab": "",
                    "status": "fail",
                    "message": message,
                }
            )
            print(f"  FAIL: {message}", flush=True)

    csv_path = results_dir / f"torchspin_vs_easyspin_timings_{run_date}.csv"
    md_path = results_dir / f"torchspin_vs_easyspin_timings_{run_date}.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
