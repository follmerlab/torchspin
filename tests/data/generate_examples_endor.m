% generate_examples_endor.m
% Generates MATLAB EasySpin reference data for torchspin endor examples.
%
% Usage:
%   cd <repo_root>
%   addpath easyspin
%   run tests/data/generate_examples_endor.m
%
% Outputs:
%   example_endorsimple.mat  - endorsimple.py validation

clear;
outdir = fileparts(mfilename('fullpath'));

fprintf('Generating endor example reference data...\n');

%% ---- endorsimple: 1H ENDOR powder, A=[5,5,20] MHz ----
Sys.S = 0.5; Sys.g = 2.0;
Sys.Nucs = '1H'; Sys.A = [5 5 20];  % MHz (axial)
Sys.lwEndor = 0.2;   % ENDOR linewidth MHz

% B0 = 330 mT
B0_mT = 330;
nu_H = larmorfrq('1H', B0_mT*1e-3);   % MHz at this field

Exp.Field = B0_mT;
Exp.mwFreq = 9.5;
Exp.Range = [nu_H - 15 nu_H + 15];   % MHz ENDOR range
Exp.nPoints = 1024;

Opt.GridSize = 40; Opt.GridSymmetry = 'Ci';
[nu_axis, y_endor] = salt(Sys, Exp, Opt);
save(fullfile(outdir, 'example_endorsimple.mat'), 'nu_axis', 'y_endor', 'nu_H', 'B0_mT');
fprintf('  Saved: example_endorsimple.mat\n');
clear Sys Exp Opt nu_axis y_endor nu_H B0_mT;

fprintf('Done. All endor reference files written to:\n  %s\n', outdir);
