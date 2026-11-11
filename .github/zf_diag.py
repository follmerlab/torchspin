"""Diagnostic for the flaky zero-field orientations test (temporary)."""
import os, platform, numpy as np, torch
from torchspin import SpinSystem, Experiment, Options, pepper
print("cpu:", platform.processor(), os.cpu_count(), "| torch", torch.__version__, "threads", torch.get_num_threads(),
      "| numpy", np.__version__, "| cpu cap", torch.backends.cpu.get_cpu_capability())
try:
    from numpy._core._multiarray_umath import __cpu_features__ as f
    print("avx512f", f.get("AVX512F"), "avx512_skx", f.get("AVX512_SKX"))
except Exception as e:
    print("no cpu features", e)
n = 4000; i = np.arange(n) + 0.5
ori = np.stack([(np.pi * (1 + 5 ** 0.5) * i) % (2 * np.pi), np.arccos(1 - 2 * i / n)], 1)
sy = SpinSystem(S=1.0, g=[2, 2, 2], D=[-700, -1300, 2000], lw=[0, 300.0])
def s(y):
    y = y.double(); return f"max {float(y.max()):.10e} argmax {int(y.argmax())} sum {float(y.sum()):.10e}"
for rep in range(3):
    a = pepper(sy, Experiment(Field=0.0, mwRange=[0, 5], nPoints=501), Options(ZeroField=True, GridSize=20))[1]
    b = pepper(sy, Experiment(Field=0.0, mwRange=[0, 5], nPoints=501, Orientations=ori), Options(ZeroField=True))[1]
    print(f"rep {rep} dev {float((a - b).abs().max() / a.abs().max()):.6f}")
    print("   powder ", s(a)); print("   orient ", s(b))
from torchspin import ham
from torchspin._linalg import eigh
H0, mx, my, mz = ham(sy)
E, V = eigh(H0)
print("E0", [f"{float(e):.6f}" for e in E.real])
