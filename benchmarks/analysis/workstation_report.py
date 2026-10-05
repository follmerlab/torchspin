#!/usr/bin/env python3
"""Summarize a workstation campaign directory (benchmarks/cluster/run_workstation.sh) as Markdown.

    python benchmarks/analysis/workstation_report.py benchmarks/results/workstation_20260902 [--after DIR]

Prints the MATLAB vs torchspin table (CPU best thread count, GPU, multi-process fit
loop) and, with --after, the before/after columns for a second results directory.
"""
import argparse, json, re, sys
from pathlib import Path


def _load(d, name):
    p = Path(d) / name
    return json.load(open(p)) if p.exists() else None


def _seconds(d, name):
    j = _load(d, name)
    return (j or {}).get('seconds', {})


def _profile_hotspots(path, n=6):
    """Top-n cumulative entries (excluding wrappers) per workload from a cProfile dump."""
    if not Path(path).exists():
        return {}
    out, cur = {}, None
    for line in open(path):
        m = re.match(r'=== (\S+) ===', line)
        if m:
            cur = m.group(1); out[cur] = []; continue
        m = re.match(r'\s*(\d+)(?:/\d+)?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$', line)
        if m and cur is not None:
            calls, tott, _, cumt, _, where = m.groups()
            where = where.strip()
            if any(k in where for k in ('run_workloads', '<lambda>', 'workloads.py', 'builtins.exec', '{method', 'pepper.py:', 'cardamom.py:', 'chili.py:')) and not any(
                    k in where for k in ('_projecttriangles', '_gaussian_bins', '_interp', 'resfields', 'strain', 'propagate', 'rotate', 'liouv', 'eigh', 'eigvalsh')):
                continue
            if len(out[cur]) < n:
                out[cur].append((int(calls), float(cumt), where))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--after', default='')
    a = ap.parse_args()
    d = a.dir
    threads = {th: _seconds(d, f'cpu_threads{th}.json') for th in (1, 8, 32)}
    gpu = _seconds(d, 'gpu.json')
    procs = {}
    for np_ in (4, 16, 64):
        procs.update(_seconds(d, f'cpu_procs{np_}.json'))
    mat = _seconds(d, 'matlab_workloads.json')
    names = list(threads[1] or threads[8] or gpu or mat)
    print('| Workload | MATLAB | torchspin 1 thr | 8 thr | 32 thr | best CPU | GPU (RTX 4090) | MATLAB/best CPU | MATLAB/GPU |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    f = lambda v: '' if v is None else f'{v:.2f}'
    for n in names:
        t = {th: threads[th].get(n) for th in threads}
        best = min([v for v in t.values() if v is not None], default=None)
        g = gpu.get(n); m = mat.get(n)
        r1 = f'{m / best:.1f}×' if (m and best) else ''
        r2 = f'{m / g:.1f}×' if (m and g) else ''
        print(f'| {n} | {f(m)} | {f(t[1])} | {f(t[8])} | {f(t[32])} | {f(best)} | {f(g)} | {r1} | {r2} |')
    if procs:
        print('\nFit loop (20 forward calls) with worker processes (1 thread each):\n')
        print('| processes | 1 (serial, 1 thr) | 4 | 16 | 64 | MATLAB (serial) |')
        print('|---|---:|---:|---:|---:|---:|')
        s1 = threads[1].get('pepper_fit_loop_20')
        print('| seconds | ' + f(s1) + ' | ' + ' | '.join(f(procs.get(f'pepper_fit_loop_20_procs{k}')) for k in (4, 16, 64)) + ' | ' + f(mat.get('pepper_fit_loop_20')) + ' |')
    if a.after:
        print(f'\nBefore/after ({d} → {a.after}), best CPU thread count and GPU:\n')
        print('| Workload | before CPU | after CPU | speedup | before GPU | after GPU | speedup |')
        print('|---|---:|---:|---:|---:|---:|---:|')
        after_cpu = {}
        for th in (1, 8, 32):
            for k, v in _seconds(a.after, f'cpu_threads{th}.json').items():
                after_cpu[k] = min(v, after_cpu.get(k, v))
        after_gpu = _seconds(a.after, 'gpu.json')
        for n in names:
            b = min([threads[th].get(n) for th in threads if threads[th].get(n) is not None], default=None)
            af = after_cpu.get(n); bg = gpu.get(n); ag = after_gpu.get(n)
            print(f'| {n} | {f(b)} | {f(af)} | {f"{b/af:.1f}×" if b and af else ""} | {f(bg)} | {f(ag)} | {f"{bg/ag:.1f}×" if bg and ag else ""} |')
    for prof in ('cpu_profile.json.profile.txt', 'gpu_profile.json.profile.txt'):
        hs = _profile_hotspots(Path(d) / prof)
        if hs:
            print(f'\nHot spots ({prof.split(".")[0]}, cumulative seconds):\n')
            for w, rows in hs.items():
                print(f'- **{w}**: ' + '; '.join(f'`{where.split("(")[-1].rstrip(")")}` {cum:.1f} s ({calls} calls)' for calls, cum, where in rows))
    return 0


if __name__ == '__main__':
    sys.exit(main())
