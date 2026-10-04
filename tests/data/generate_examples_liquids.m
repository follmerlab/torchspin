% generate_examples_liquids.m
% Generates MATLAB EasySpin reference data for torchspin liquids examples.
%
% Usage:
%   cd <repo_root>
%   addpath easyspin
%   run tests/data/generate_examples_liquids.m
%
% Outputs:
%   example_fremysalt.mat    - fremysalt.py validation
%   example_biphenyl.mat     - biphenyl.py validation

clear;
outdir = fileparts(mfilename('fullpath'));

fprintf('Generating liquids example reference data...\n');

%% ---- Fremy's salt: S=1/2, 14N, fast-motion ----
% A(14N) = 13.09 G in solution
A_G = 13.09; A_MHz = A_G * 2.80250;  % approx for g=2
Sys.S = 0.5; Sys.g = 2.0055;
Sys.Nucs = '14N'; Sys.A = A_MHz;  % isotropic
Sys.lw = [0 0.15];
Exp.mwFreq = 9.5; Exp.Range = [335 342]; Exp.nPoints = 2048; Exp.Harmonic = 1;
[~, y_fremy] = garlic(Sys, Exp);
save(fullfile(outdir, 'example_fremysalt.mat'), 'y_fremy');
fprintf('  Saved: example_fremysalt.mat\n');
clear Sys Exp y_fremy A_G A_MHz;

%% ---- Biphenyl radical anion: 4+4 protons ----
% 4 × 7.43 MHz + 4 × 2.74 MHz protons
Sys.S = 0.5; Sys.g = 2.0027;
Sys.Nucs = '1H,1H,1H,1H,1H,1H,1H,1H';
Sys.A = [7.43 7.43 7.43 7.43 2.74 2.74 2.74 2.74];  % isotropic (8 values)
Sys.lw = [0 0.1];
Exp.mwFreq = 9.5; Exp.Range = [336 342]; Exp.nPoints = 2048; Exp.Harmonic = 1;
[~, y_biphenyl] = garlic(Sys, Exp);
save(fullfile(outdir, 'example_biphenyl.mat'), 'y_biphenyl');
fprintf('  Saved: example_biphenyl.mat\n');
clear Sys Exp y_biphenyl;

fprintf('Done. All liquids reference files written to:\n  %s\n', outdir);
