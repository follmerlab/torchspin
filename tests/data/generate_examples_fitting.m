% generate_examples_fitting.m
% Generates MATLAB EasySpin reference data for torchspin fitting examples.
%
% These files store the "true" spectrum (no noise) used as fitting targets
% so that the Python esfit examples can be validated against a known good spectrum.
%
% Usage:
%   cd <repo_root>
%   addpath easyspin
%   run tests/data/generate_examples_fitting.m
%
% Outputs:
%   example_basicfit_clean.mat      - noiseless target for basicfit.py
%   example_multicomponents_clean.mat - noiseless targets for multicomponents.py

clear;
outdir = fileparts(mfilename('fullpath'));

fprintf('Generating fitting example reference data...\n');

%% ---- basicfit: S=1/2, 1H, clean spectrum ----
Sys.S = 0.5; Sys.g = [2.00 2.10 2.20];
Sys.Nucs = '1H'; Sys.A = [120 50 78];  % MHz
Sys.lw = [1 0];
Exp.mwFreq = 10; Exp.Range = [300 380]; Exp.nPoints = 1024; Exp.Harmonic = 1;
Opt.GridSize = 31; Opt.GridSymmetry = 'C2h';
[~, y_basicfit] = pepper(Sys, Exp, Opt);
save(fullfile(outdir, 'example_basicfit_clean.mat'), 'y_basicfit');
fprintf('  Saved: example_basicfit_clean.mat\n');
clear Sys Exp Opt y_basicfit;

%% ---- multicomponents: two S=1/2 species ----
% Component 1: g=[2, 2, 2.2], weight=1
Sys1.S = 0.5; Sys1.g = [2.0 2.0 2.2]; Sys1.lw = [1 0];
% Component 2: g=[2.1, 2.1, 2.15], weight=0.3
Sys2.S = 0.5; Sys2.g = [2.1 2.1 2.15]; Sys2.lw = [1 0];
Exp.mwFreq = 9.5; Exp.Range = [280 370]; Exp.nPoints = 1024; Exp.Harmonic = 1;
Opt.GridSize = 31; Opt.GridSymmetry = 'Ci';
[~, y1] = pepper(Sys1, Exp, Opt);
[~, y2] = pepper(Sys2, Exp, Opt);
y_multi_clean = y1 + 0.3 * y2;
save(fullfile(outdir, 'example_multicomponents_clean.mat'), 'y1', 'y2', 'y_multi_clean');
fprintf('  Saved: example_multicomponents_clean.mat\n');
clear Sys1 Sys2 Exp Opt y1 y2 y_multi_clean;

fprintf('Done. All fitting reference files written to:\n  %s\n', outdir);
