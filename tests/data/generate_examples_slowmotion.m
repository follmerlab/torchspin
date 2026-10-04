% generate_examples_slowmotion.m
% Generates MATLAB EasySpin reference data for torchspin slowmotion examples.
%
% Usage:
%   cd <repo_root>
%   addpath easyspin
%   run tests/data/generate_examples_slowmotion.m
%
% Outputs:
%   example_nitroxide_basic.mat  - nitroxide_basic.py validation
%   example_nitroxide_tcorr.mat  - nitroxide_tcorr.py validation

clear;
outdir = fileparts(mfilename('fullpath'));

fprintf('Generating slowmotion example reference data...\n');

%% ---- nitroxide basic: tcorr = 3 ns ----
Sys.S = 0.5;
Sys.g = [2.0089 2.0058 2.0021];
Sys.Nucs = '14N';
Sys.A = [16 16 100];   % MHz
Sys.tcorr = 3e-9;
Sys.lw = [0 0.05];
Exp.mwFreq = 9.5; Exp.Range = [328 352]; Exp.nPoints = 2048; Exp.Harmonic = 1;
[~, y_basic] = chili(Sys, Exp);
save(fullfile(outdir, 'example_nitroxide_basic.mat'), 'y_basic');
fprintf('  Saved: example_nitroxide_basic.mat\n');
clear Sys Exp y_basic;

%% ---- nitroxide tcorr: multiple correlation times ----
tcorr_list = [1e-7 1e-8 1e-9 1e-10];
Sys.S = 0.5; Sys.g = [2.0089 2.0058 2.0021];
Sys.Nucs = '14N'; Sys.A = [16 16 100]; Sys.lw = [0 0.05];
Exp.mwFreq = 9.5; Exp.Range = [328 352]; Exp.nPoints = 2048; Exp.Harmonic = 1;
y_all = zeros(length(tcorr_list), 2048);
for k = 1:length(tcorr_list)
    Sys.tcorr = tcorr_list(k);
    [~, y] = chili(Sys, Exp);
    y_all(k,:) = y;
end
save(fullfile(outdir, 'example_nitroxide_tcorr.mat'), 'tcorr_list', 'y_all');
fprintf('  Saved: example_nitroxide_tcorr.mat\n');
clear Sys Exp tcorr_list y y_all k;

fprintf('Done. All slowmotion reference files written to:\n  %s\n', outdir);
