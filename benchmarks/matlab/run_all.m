% RUN_ALL  Run all MATLAB EasySpin benchmarks for the torchspin manuscript.
%
% Mirrors the test cases in benchmarks/common/spin_systems.py.
% Output: benchmarks/results/matlab_timings.mat and matlab_outputs.mat
%
% Usage:
%   matlab -batch "addpath('easyspin'); cd benchmarks/matlab; run_all"

clear all;

resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

results = struct();
outputs = struct();

n_runs = 5;
n_warmup = 1;

% =====================================================================
% pepper
% =====================================================================

% nitroxide_xband
clear Sys Exp Opt;
Sys.S = 1/2; Sys.g = [2.009 2.006 2.002]; Sys.Nucs = '14N';
Sys.A = [10 10 95]; Sys.lw = [1 0];
Exp.mwFreq = 9.5; Exp.Range = [330 350]; Exp.nPoints = 1024; Exp.Harmonic = 1;
Opt.GridSize = 50; Opt.GridSymmetry = 'Ci'; Opt.Verbosity = 0;
[B, spc, t] = bench_run(@() pepper(Sys, Exp, Opt), n_runs, n_warmup);
results.pepper.nitroxide_xband.best_s = min(t);
results.pepper.nitroxide_xband.median_s = median(t);
results.pepper.nitroxide_xband.all_s = t;
outputs.pepper__nitroxide_xband.B = B; outputs.pepper__nitroxide_xband.spc = spc;
fprintf('  pepper/nitroxide_xband: best=%.1fms\n', min(t)*1000);

% cuII_rhombic
clear Sys Exp Opt;
Sys.S = 1/2; Sys.g = [2.05 2.10 2.30]; Sys.lw = [2 0];
Exp.mwFreq = 9.5; Exp.Range = [280 380]; Exp.nPoints = 1024; Exp.Harmonic = 1;
Opt.GridSize = 50; Opt.GridSymmetry = 'Ci'; Opt.Verbosity = 0;
[B, spc, t] = bench_run(@() pepper(Sys, Exp, Opt), n_runs, n_warmup);
results.pepper.cuII_rhombic.best_s = min(t);
results.pepper.cuII_rhombic.median_s = median(t);
results.pepper.cuII_rhombic.all_s = t;
outputs.pepper__cuII_rhombic.B = B; outputs.pepper__cuII_rhombic.spc = spc;
fprintf('  pepper/cuII_rhombic: best=%.1fms\n', min(t)*1000);

% organic_radical_gstrain
clear Sys Exp Opt;
Sys.S = 1/2; Sys.g = [2.0104 2.0074 2.0026];
Sys.gStrain = [0.001 0.0008 0.0005]; Sys.lw = [0 0];
Exp.mwFreq = 9.5; Exp.Range = [332 342]; Exp.nPoints = 1024; Exp.Harmonic = 1;
Opt.GridSize = 31; Opt.Verbosity = 0;
[B, spc, t] = bench_run(@() pepper(Sys, Exp, Opt), n_runs, n_warmup);
results.pepper.organic_radical_gstrain.best_s = min(t);
results.pepper.organic_radical_gstrain.median_s = median(t);
results.pepper.organic_radical_gstrain.all_s = t;
outputs.pepper__organic_radical_gstrain.B = B; outputs.pepper__organic_radical_gstrain.spc = spc;
fprintf('  pepper/organic_radical_gstrain: best=%.1fms\n', min(t)*1000);

% triplet_zfs
clear Sys Exp Opt;
Sys.S = 1; Sys.g = [2 2 2]; Sys.D = [200 200 -400]; Sys.lw = [1 0];
Exp.mwFreq = 9.5; Exp.Range = [300 380]; Exp.nPoints = 1024; Exp.Harmonic = 0;
Opt.GridSize = 50; Opt.Verbosity = 0;
[B, spc, t] = bench_run(@() pepper(Sys, Exp, Opt), n_runs, n_warmup);
results.pepper.triplet_zfs.best_s = min(t);
results.pepper.triplet_zfs.median_s = median(t);
results.pepper.triplet_zfs.all_s = t;
outputs.pepper__triplet_zfs.B = B; outputs.pepper__triplet_zfs.spc = spc;
fprintf('  pepper/triplet_zfs: best=%.1fms\n', min(t)*1000);

% =====================================================================
% garlic
% =====================================================================

clear Sys Exp;
Sys.S = 1/2; Sys.g = [2.009 2.006 2.002]; Sys.Nucs = '14N';
Sys.A = [16 16 95]; Sys.tcorr = 1e-10; Sys.lw = [0.1 0];
Exp.mwFreq = 9.5; Exp.Range = [336 342]; Exp.nPoints = 1024; Exp.Harmonic = 1;
[B, spc, t] = bench_run(@() garlic(Sys, Exp), n_runs, n_warmup);
results.garlic.nitroxide_fastmotion.best_s = min(t);
results.garlic.nitroxide_fastmotion.median_s = median(t);
results.garlic.nitroxide_fastmotion.all_s = t;
outputs.garlic__nitroxide_fastmotion.B = B; outputs.garlic__nitroxide_fastmotion.spc = spc;
fprintf('  garlic/nitroxide_fastmotion: best=%.1fms\n', min(t)*1000);

clear Sys Exp;
Sys.S = 1/2; Sys.g = [2.0058 2.0061 2.0022]; Sys.Nucs = '14N,1H,1H';
Sys.A = [16 16 95; 5 5 5; 5 5 5]; Sys.lw = [0.2 0];
Exp.mwFreq = 9.5; Exp.Range = [330 348]; Exp.nPoints = 1024; Exp.Harmonic = 1;
[B, spc, t] = bench_run(@() garlic(Sys, Exp), n_runs, n_warmup);
results.garlic.multinuc_pattern.best_s = min(t);
results.garlic.multinuc_pattern.median_s = median(t);
results.garlic.multinuc_pattern.all_s = t;
outputs.garlic__multinuc_pattern.B = B; outputs.garlic__multinuc_pattern.spc = spc;
fprintf('  garlic/multinuc_pattern: best=%.1fms\n', min(t)*1000);

% =====================================================================
% chili
% =====================================================================

clear Sys Exp Opt;
Sys.S = 1/2; Sys.g = [2.009 2.006 2.002]; Sys.Nucs = '14N';
Sys.A = [16 16 95]; Sys.tcorr = 1e-9; Sys.lw = [0 0];
Exp.mwFreq = 9.5; Exp.Range = [332 342]; Exp.nPoints = 256; Exp.Harmonic = 0;
Opt.LLMK = [14 7 2 6]; Opt.MaxIter = 200;
[B, spc, t] = bench_run(@() chili(Sys, Exp, Opt), 3, 1);
results.chili.nitroxide_tcorr1ns.best_s = min(t);
results.chili.nitroxide_tcorr1ns.median_s = median(t);
results.chili.nitroxide_tcorr1ns.all_s = t;
outputs.chili__nitroxide_tcorr1ns.B = B; outputs.chili__nitroxide_tcorr1ns.spc = spc;
fprintf('  chili/nitroxide_tcorr1ns: best=%.1fms\n', min(t)*1000);

Sys.tcorr = 1e-8;
Opt.MaxIter = 300;
[B, spc, t] = bench_run(@() chili(Sys, Exp, Opt), 3, 1);
results.chili.nitroxide_tcorr10ns.best_s = min(t);
results.chili.nitroxide_tcorr10ns.median_s = median(t);
results.chili.nitroxide_tcorr10ns.all_s = t;
outputs.chili__nitroxide_tcorr10ns.B = B; outputs.chili__nitroxide_tcorr10ns.spc = spc;
fprintf('  chili/nitroxide_tcorr10ns: best=%.1fms\n', min(t)*1000);

% =====================================================================
% saffron
% =====================================================================

clear Sys Exp Opt;
Sys.S = 1/2; Sys.g = [2.0023 2.0023 2.0023]; Sys.Nucs = '1H';
Sys.A = [3 3 9];
Exp.Sequence = '2pESEEM'; Exp.Field = 350; Exp.dt = 0.010;
Exp.tau = 0.0; Exp.nPoints = 256;
Opt.GridSize = 30;
[x, y, t] = bench_run(@() saffron(Sys, Exp, Opt), 3, 1);
results.saffron.x2pESEEM_1H.best_s = min(t);
results.saffron.x2pESEEM_1H.median_s = median(t);
results.saffron.x2pESEEM_1H.all_s = t;
outputs.saffron__x2pESEEM_1H.x = x; outputs.saffron__x2pESEEM_1H.y = y;
fprintf('  saffron/2pESEEM_1H: best=%.1fms\n', min(t)*1000);

clear Exp;
Exp.Sequence = '3pESEEM'; Exp.Field = 350; Exp.dt = 0.010;
Exp.tau = 0.1; Exp.nPoints = 256;
[x, y, t] = bench_run(@() saffron(Sys, Exp, Opt), 3, 1);
results.saffron.x3pESEEM_1H.best_s = min(t);
results.saffron.x3pESEEM_1H.median_s = median(t);
results.saffron.x3pESEEM_1H.all_s = t;
outputs.saffron__x3pESEEM_1H.x = x; outputs.saffron__x3pESEEM_1H.y = y;
fprintf('  saffron/3pESEEM_1H: best=%.1fms\n', min(t)*1000);

clear Exp Opt;
Exp.Sequence = 'HYSCORE'; Exp.Field = 324.9; Exp.dt = 0.050;
Exp.nPoints = [128 128]; Exp.tau = 0.08; Exp.t1 = 0.1; Exp.t2 = 0.1;
Opt.GridSize = 20;
[x, y, t] = bench_run(@() saffron(Sys, Exp, Opt), 3, 1);
results.saffron.HYSCORE_1H.best_s = min(t);
results.saffron.HYSCORE_1H.median_s = median(t);
results.saffron.HYSCORE_1H.all_s = t;
outputs.saffron__HYSCORE_1H.x = x; outputs.saffron__HYSCORE_1H.y = y;
fprintf('  saffron/HYSCORE_1H: best=%.1fms\n', min(t)*1000);

% =====================================================================
% esfit (synthetic data → fit g-tensor)
% =====================================================================

clear Sys Exp Opt;
Sys_true.S = 1/2; Sys_true.g = [2.009 2.006 2.002]; Sys_true.lw = [1 0];
Exp.mwFreq = 9.5; Exp.Range = [330 350]; Exp.nPoints = 1024; Exp.Harmonic = 1;
Opt.GridSize = 31; Opt.Verbosity = 0;
[~, y_true] = pepper(Sys_true, Exp, Opt);

% Initial guess (different g)
Sys_init.S = 1/2; Sys_init.g = [2.005 2.005 2.005]; Sys_init.lw = [1 0];
Vary.g = [0.05 0.05 0.05];
FitOpt.Verbosity = 0; FitOpt.maxiter = 200;
[x, y, t] = bench_run(@() run_esfit(y_true, Sys_init, Vary, Exp, Opt, FitOpt), 3, 1);
results.esfit.nitroxide_g_fit.best_s = min(t);
results.esfit.nitroxide_g_fit.median_s = median(t);
results.esfit.nitroxide_g_fit.all_s = t;
outputs.esfit__nitroxide_g_fit.fit_struct = x;
fprintf('  esfit/nitroxide_g_fit: best=%.1fms\n', min(t)*1000);

% =====================================================================
% Save
% =====================================================================

meta.matlab_version = version;
meta.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
% Store the key timings per simulator/case
save(fullfile(resultsDir, 'matlab_timings.mat'), 'results', 'meta');
save(fullfile(resultsDir, 'matlab_outputs.mat'), 'outputs', 'meta');

fprintf('\nDone. Wrote:\n  %s\n  %s\n', ...
  fullfile(resultsDir, 'matlab_timings.mat'), ...
  fullfile(resultsDir, 'matlab_outputs.mat'));

function [fit, placeholder] = run_esfit(data, Sys0, Vary, Exp, SimOpt, FitOpt)
% Adapt EasySpin 6's cell-array esfit API to bench_run's two-output contract.
fit = esfit(data, @pepper, {Sys0, Exp, SimOpt}, {Vary}, FitOpt);
placeholder = [];
end
