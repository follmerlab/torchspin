% generate_examples_solidstate.m
% Generates MATLAB EasySpin reference data for torchspin solidstate examples.
%
% Usage:
%   cd <repo_root>
%   addpath easyspin
%   run tests/data/generate_examples_solidstate.m
%
% Outputs (all saved in same directory as this script):
%   example_broaden.mat        - broaden.py validation
%   example_gstrain.mat        - gstrain.py validation
%   example_iron_highspin.mat  - iron_highspin.py validation
%   example_triplet_naphthalene.mat  - triplet_naphthalene.py validation
%   example_temperature.mat    - temperature.py validation

clear;
outdir = fileparts(mfilename('fullpath'));
CM1_TO_MHZ = 29979.2458;

fprintf('Generating solidstate example reference data...\n');

%% ---- broaden: three broadening mechanisms ----
Sys.S = 0.5; Sys.g = [1.9 2.01 2.3];
Exp.mwFreq = 9.5; Exp.Range = [280 370]; Exp.nPoints = 1024; Exp.Harmonic = 0;
Opt.GridSize = 50; Opt.GridSymmetry = 'Ci';

% HStrain
Sys.lw = 0; Sys.HStrain = [110 40 50];
[~, y1] = pepper(Sys, Exp, Opt);

% Gaussian lw
Sys.HStrain = [0 0 0]; Sys.lw = 5.0;
[~, y2] = pepper(Sys, Exp, Opt);

% gStrain
Sys.lw = 0; Sys.gStrain = [0.05 0.01 0.01];
[~, y3] = pepper(Sys, Exp, Opt);

save(fullfile(outdir, 'example_broaden.mat'), 'y1', 'y2', 'y3');
fprintf('  Saved: example_broaden.mat\n');
clear Sys Exp Opt y1 y2 y3;

%% ---- gstrain: multiple frequencies ----
Sys.S = 0.5; Sys.g = [2.0104 2.0074 2.0026]; Sys.gStrain = [0.001 0.0008 0.0005];
Exp.Harmonic = 0; Exp.nPoints = 512;
Opt.GridSize = 50; Opt.GridSymmetry = 'Ci';

freqs_GHz = [3 9.5 35 95 350];
ranges_mT = [102 112; 334 344; 1240 1255; 3372 3395; 12425 12505];
y_all = zeros(length(freqs_GHz), 512);
for k = 1:length(freqs_GHz)
    Exp.mwFreq = freqs_GHz(k);
    Exp.Range = ranges_mT(k,:);
    [~, y] = pepper(Sys, Exp, Opt);
    y_all(k,:) = y;
end
save(fullfile(outdir, 'example_gstrain.mat'), 'freqs_GHz', 'ranges_mT', 'y_all');
fprintf('  Saved: example_gstrain.mat\n');
clear Sys Exp Opt y y_all freqs_GHz ranges_mT k;

%% ---- iron_highspin: S=5/2 Fe(III) ----
D_cm = 10.0; E_cm = 0.5;
D_MHz = D_cm * CM1_TO_MHZ;
E_MHz = E_cm * CM1_TO_MHZ;
Dxx = -D_MHz/3 + E_MHz; Dyy = -D_MHz/3 - E_MHz; Dzz = 2*D_MHz/3;

Fe_full.S = 2.5; Fe_full.g = [2 2 2];
Fe_full.D = [Dxx Dyy Dzz];  % [Dxx Dyy Dzz] in MHz
Fe_full.lw = [5.0 0];
Fe_eff.S = 0.5; Fe_eff.g = [7.13 4.78 1.919]; Fe_eff.lw = [5.0 0];

Exp.mwFreq = 9.5; Exp.Range = [0 700]; Exp.nPoints = 5000;
Exp.Harmonic = 0; Exp.Temperature = 10;

Opt.GridSize = 31; Opt.GridSymmetry = 'Ci';
[~, y_full] = pepper(Fe_full, Exp, Opt);
[~, y_eff]  = pepper(Fe_eff,  Exp, Opt);
save(fullfile(outdir, 'example_iron_highspin.mat'), 'y_full', 'y_eff');
fprintf('  Saved: example_iron_highspin.mat\n');
clear Fe_full Fe_eff Exp Opt y_full y_eff D_cm E_cm D_MHz E_MHz Dxx Dyy Dzz;

%% ---- triplet_naphthalene: S=1 triplet ----
D_cm = 0.1003; E_cm = 0.0137;
D_MHz = D_cm * CM1_TO_MHZ;
E_MHz = E_cm * CM1_TO_MHZ;
Dxx = -D_MHz/3 + E_MHz; Dyy = -D_MHz/3 - E_MHz; Dzz = 2*D_MHz/3;

Sys.S = 1; Sys.g = [2.003 2.003 2.003];
Sys.D = [Dxx Dyy Dzz];
Sys.lw = [6.0 0];
Exp.mwFreq = 9.5; Exp.Range = [0 500]; Exp.nPoints = 1024; Exp.Harmonic = 1;
Opt.GridSize = 50; Opt.GridSymmetry = 'Ci';
[~, y_naph] = pepper(Sys, Exp, Opt);
save(fullfile(outdir, 'example_triplet_naphthalene.mat'), 'y_naph');
fprintf('  Saved: example_triplet_naphthalene.mat\n');
clear Sys Exp Opt y_naph D_cm E_cm D_MHz E_MHz Dxx Dyy Dzz;

%% ---- temperature: S=1 at multiple temperatures ----
Sys.S = 1; Sys.g = [2 2 2]; Sys.D = [200 200 -400]; Sys.lw = [1 0];
Exp.mwFreq = 9.5; Exp.Range = [300 380]; Exp.nPoints = 1024; Exp.Harmonic = 0;
Opt.GridSize = 40; Opt.GridSymmetry = 'Ci';

Temps = [80 40 20 10 5 2];
y_temps = zeros(length(Temps), 1024);
for k = 1:length(Temps)
    Exp.Temperature = Temps(k);
    [~, y] = pepper(Sys, Exp, Opt);
    y_temps(k,:) = y;
end
save(fullfile(outdir, 'example_temperature.mat'), 'Temps', 'y_temps');
fprintf('  Saved: example_temperature.mat\n');
clear Sys Exp Opt Temps y y_temps k;

fprintf('Done. All solidstate reference files written to:\n  %s\n', outdir);
