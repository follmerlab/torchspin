"""Compare torchspin example outputs against MATLAB EasySpin references.

This script validates the subset of examples/ for which MATLAB
reference files are generated under tests/data/.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DATA_DIR = REPO_ROOT / "tests" / "data"
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ComparisonResult:
    example: str
    status: str
    metric: str
    value: float | str
    threshold: float | str
    note: str = ""


def _import_example(example_key: str):
    subdir, name = example_key.split("/")
    path = EXAMPLES_ROOT / subdir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    peak = np.max(np.abs(x))
    return x if peak == 0 else x / peak


def _match_length(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape == b.shape:
        return a, b
    n = max(len(a), len(b))
    grid = np.linspace(0.0, 1.0, n)
    a_rs = np.interp(grid, np.linspace(0.0, 1.0, len(a)), a)
    b_rs = np.interp(grid, np.linspace(0.0, 1.0, len(b)), b)
    return a_rs, b_rs


def _cosine_similarity(a, b) -> float:
    a, b = _match_length(a, b)
    a = _norm(a)
    b = _norm(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def _relative_l2(a, b) -> float:
    a, b = _match_length(a, b)
    a = _norm(a)
    b = _norm(b)
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def _relative_l2_raw(a, b) -> float:
    a, b = _match_length(a, b)
    denom = np.linalg.norm(b.ravel())
    if denom == 0:
        return float(np.linalg.norm((a - b).ravel()))
    return float(np.linalg.norm((a - b).ravel()) / denom)


def _assert_axis_close(a, b, atol=1e-6):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        raise AssertionError(f"axis shape mismatch: {a.shape} vs {b.shape}")
    if not np.allclose(a, b, atol=atol, rtol=0):
        raise AssertionError(f"axis mismatch: max |Δ|={np.max(np.abs(a-b)):.3e}")


def compare_solidstate_broaden():
    ref = loadmat(DATA_DIR / "example_broaden.mat", squeeze_me=True)
    _B, spc1, spc2, spc3 = _import_example("solidstate/broaden").run()
    checks = [
        ("hstrain", spc1, ref["y1"]),
        ("gaussian", spc2, ref["y2"]),
        ("gstrain", spc3, ref["y3"]),
    ]
    worst = min(_cosine_similarity(py, mat) for _, py, mat in checks)
    return ComparisonResult("solidstate/broaden", "pass" if worst >= 0.995 else "fail", "min_cosine", worst, 0.995)


def compare_solidstate_gstrain():
    ref = loadmat(DATA_DIR / "example_gstrain.mat", squeeze_me=True)
    results = _import_example("solidstate/gstrain").run()[0]
    cosines = []
    for i, (_B, spc_no, spc_gs) in enumerate(results):
        cosines.append(_cosine_similarity(spc_gs, ref["y_all"][i]))
    worst = min(cosines)
    return ComparisonResult("solidstate/gstrain", "pass" if worst >= 0.995 else "fail", "min_cosine", worst, 0.995)


def compare_solidstate_iron_highspin():
    ref = loadmat(DATA_DIR / "example_iron_highspin.mat", squeeze_me=True)
    _B, y_full, y_eff, _D, _E = _import_example("solidstate/iron_highspin").run()
    c1 = _cosine_similarity(y_full, ref["y_full"])
    c2 = _cosine_similarity(y_eff, ref["y_eff"])
    worst = min(c1, c2)
    return ComparisonResult("solidstate/iron_highspin", "pass" if worst >= 0.97 else "fail", "min_cosine", worst, 0.97)


def compare_solidstate_triplet_naphthalene():
    ref = loadmat(DATA_DIR / "example_triplet_naphthalene.mat", squeeze_me=True)
    _B, spc, _D, _E = _import_example("solidstate/triplet_naphthalene").run()
    cs = _cosine_similarity(spc, ref["y_naph"])
    return ComparisonResult("solidstate/triplet_naphthalene", "pass" if cs >= 0.995 else "fail", "cosine", cs, 0.995)


def compare_solidstate_temperature():
    ref = loadmat(DATA_DIR / "example_temperature.mat", squeeze_me=True)
    _B, spectra, temps = _import_example("solidstate/temperature").run()
    cosines = []
    for py, mat in zip(spectra, ref["y_temps"]):
        cosines.append(_cosine_similarity(py, mat))
    worst = min(cosines)
    note = f"{len(temps)} temperatures"
    return ComparisonResult("solidstate/temperature", "pass" if worst >= 0.99 else "fail", "min_cosine", worst, 0.99, note)


def compare_liquids_fremysalt():
    ref = loadmat(DATA_DIR / "example_fremysalt.mat", squeeze_me=True)
    _B, spc, _A_G, _A_MHz = _import_example("liquids/fremysalt").run()
    cs = _cosine_similarity(spc, ref["y_fremy"])
    return ComparisonResult("liquids/fremysalt", "pass" if cs >= 0.998 else "fail", "cosine", cs, 0.998)


def compare_liquids_biphenyl():
    ref = loadmat(DATA_DIR / "example_biphenyl.mat", squeeze_me=True)
    _B, spc = _import_example("liquids/biphenyl").run()
    cs = _cosine_similarity(spc, ref["y_biphenyl"])
    return ComparisonResult("liquids/biphenyl", "pass" if cs >= 0.998 else "fail", "cosine", cs, 0.998)


def compare_magnetometry_curie_law():
    ref = loadmat(DATA_DIR / "example_curie_law.mat", squeeze_me=True)
    T_arr, chi_cm3, _chi_curie, _chiT, _C = _import_example("magnetometry/curie_law").run()
    chi_si_py = np.asarray(chi_cm3) / 1e6
    chi_interp = np.interp(ref["T_arr"], T_arr, chi_si_py)
    rel = _relative_l2_raw(chi_interp, ref["chi_mol"])
    return ComparisonResult("magnetometry/curie_law", "pass" if rel <= 0.02 else "fail", "relative_l2", rel, 0.02)


def compare_magnetometry_squid_basic():
    ref = loadmat(DATA_DIR / "example_squid_basic.mat", squeeze_me=True)
    B_arr, curves, _T = _import_example("magnetometry/squid_basic").run()
    M_all = np.vstack([np.interp(ref["B_arr"], B_arr, np.asarray(M)) for M, _label, _S in curves])
    rel = _relative_l2_raw(M_all, ref["M_all"])
    return ComparisonResult("magnetometry/squid_basic", "pass" if rel <= 0.02 else "fail", "relative_l2", rel, 0.02)


def compare_magnetometry_magnetization_tempdep():
    ref = loadmat(DATA_DIR / "example_magnetization_tempdep.mat", squeeze_me=True)
    T_arr, results = _import_example("magnetometry/magnetization_tempdep").run()
    mu_eff_all = np.vstack([np.interp(ref["T_arr"], T_arr, np.asarray(mu_eff)) for mu_eff, _label, _mu in results])
    # Reconstruct MATLAB mu_eff from chi(T): mu_eff = 2.828*sqrt(chi_cm3*T)
    chi_cm3 = np.asarray(ref["chi_all"]) * 1e6
    mu_eff_ref = 2.828 * np.sqrt(chi_cm3 * np.asarray(ref["T_arr"]))
    rel = _relative_l2_raw(mu_eff_all, mu_eff_ref)
    return ComparisonResult("magnetometry/magnetization_tempdep", "pass" if rel <= 0.03 else "fail", "relative_l2", rel, 0.03)


def compare_endor_endorsimple():
    ref = loadmat(DATA_DIR / "example_endorsimple.mat", squeeze_me=True)
    freq, spc, nu_H = _import_example("endor/endorsimple").run()
    _assert_axis_close(freq, ref["nu_axis"], atol=1e-6)
    cs = _cosine_similarity(spc, ref["y_endor"])
    nu_err = abs(float(nu_H) - float(ref["nu_H"]))
    status = "pass" if cs >= 0.995 and nu_err <= 1e-3 else "fail"
    return ComparisonResult("endor/endorsimple", status, "cosine", cs, 0.995, f"|Δnu_H|={nu_err:.3e}")


def compare_fitting_basicfit():
    ref = loadmat(DATA_DIR / "example_basicfit_clean.mat", squeeze_me=True)
    _B, _spc_exp, result = _import_example("fitting/basicfit").run()
    cs = _cosine_similarity(result.fit, ref["y_basicfit"])
    return ComparisonResult("fitting/basicfit", "pass" if cs >= 0.98 else "fail", "cosine_fit_vs_clean", cs, 0.98, "noisy fit vs clean MATLAB target")


def compare_fitting_multicomponents():
    ref = loadmat(DATA_DIR / "example_multicomponents_clean.mat", squeeze_me=True)
    _B, _spc_noisy, result = _import_example("fitting/multicomponents").run()
    cs = _cosine_similarity(result.fit, ref["y_multi_clean"])
    return ComparisonResult("fitting/multicomponents", "pass" if cs >= 0.95 else "fail", "cosine_fit_vs_clean", cs, 0.95, "noisy fit vs clean MATLAB target")


def compare_slowmotion_nitroxide_basic():
    ref = loadmat(DATA_DIR / "example_nitroxide_basic.mat", squeeze_me=True)
    _B, spc = _import_example("slowmotion/nitroxide_basic").run()
    cs = _cosine_similarity(spc, ref["y_basic"])
    return ComparisonResult("slowmotion/nitroxide_basic", "pass" if cs >= 0.99 else "fail", "cosine", cs, 0.99)


def compare_slowmotion_nitroxide_tcorr():
    ref = loadmat(DATA_DIR / "example_nitroxide_tcorr.mat", squeeze_me=True)
    results = _import_example("slowmotion/nitroxide_tcorr").run()
    cosines = []
    for i, (_B, spc, _tcorr) in enumerate(results):
        cosines.append(_cosine_similarity(spc, ref["y_all"][i]))
    worst = min(cosines)
    return ComparisonResult("slowmotion/nitroxide_tcorr", "pass" if worst >= 0.99 else "fail", "min_cosine", worst, 0.99)


COMPARISONS = [
    compare_solidstate_broaden,
    compare_solidstate_gstrain,
    compare_solidstate_iron_highspin,
    compare_solidstate_triplet_naphthalene,
    compare_solidstate_temperature,
    compare_liquids_fremysalt,
    compare_liquids_biphenyl,
    compare_magnetometry_curie_law,
    compare_magnetometry_squid_basic,
    compare_magnetometry_magnetization_tempdep,
    compare_fitting_basicfit,
    compare_fitting_multicomponents,
    compare_slowmotion_nitroxide_basic,
    compare_slowmotion_nitroxide_tcorr,
]


def main():
    missing = []
    for name in [
        "example_broaden.mat",
        "example_gstrain.mat",
        "example_iron_highspin.mat",
        "example_triplet_naphthalene.mat",
        "example_temperature.mat",
        "example_fremysalt.mat",
        "example_biphenyl.mat",
        "example_curie_law.mat",
        "example_squid_basic.mat",
        "example_magnetization_tempdep.mat",
        "example_basicfit_clean.mat",
        "example_multicomponents_clean.mat",
        "example_nitroxide_basic.mat",
        "example_nitroxide_tcorr.mat",
    ]:
        path = DATA_DIR / name
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("Missing MATLAB reference files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(2)

    results = []
    for fn in COMPARISONS:
        try:
            result = fn()
        except Exception as exc:
            name = fn.__name__.replace("compare_", "").replace("_", "/")
            result = ComparisonResult(name, "fail", "exception", str(exc), "")
        results.append(result)
        val = result.value if isinstance(result.value, str) else f"{result.value:.6f}"
        thr = result.threshold if isinstance(result.threshold, str) else f"{result.threshold:.6f}"
        tail = f" ({result.note})" if result.note else ""
        print(f"{result.status.upper():4s}  {result.example:<38}  {result.metric:<20}  {val:>10}  thresh={thr}{tail}")

    n_fail = sum(r.status != "pass" for r in results)
    print(f"\nSummary: {len(results)-n_fail} passed, {n_fail} failed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
