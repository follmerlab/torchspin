% generate_examples_magnetometry.m
% Generates MATLAB EasySpin reference data for torchspin magnetometry examples.
%
% Usage:
%   cd <repo_root>
%   addpath easyspin
%   run tests/data/generate_examples_magnetometry.m
%
% Outputs:
%   example_curie_law.mat           - curie_law.py validation
%   example_squid_basic.mat         - squid_basic.py validation
%   example_magnetization_tempdep.mat - magnetization_tempdep.py validation

clear;
outdir = fileparts(mfilename('fullpath'));

fprintf('Generating magnetometry example reference data...\n');

%% ---- Curie law: chi(T) for S=1/2 ----
Sys.S = 0.5; Sys.g = 2.0;
Exp.Field = 10;  % mT (small field)
Exp.Temperature = linspace(2, 300, 50);
[~, chi_mol] = curry(Sys, Exp);
% chi_mol in SI units (m^3/mol)
T_arr = Exp.Temperature;
save(fullfile(outdir, 'example_curie_law.mat'), 'T_arr', 'chi_mol');
fprintf('  Saved: example_curie_law.mat\n');
clear Sys Exp chi_mol T_arr;

%% ---- M(B) saturation: multiple spin values ----
T_K = 2.0;
B_arr = linspace(0, 7000, 50);  % mT
S_vals = [0.5, 1.0, 1.5, 2.5];
M_all = zeros(length(S_vals), length(B_arr));
for k = 1:length(S_vals)
    Sys.S = S_vals(k); Sys.g = 2.0;
    Exp.Temperature = T_K;
    Exp.Field = B_arr;
    [muz, ~] = curry(Sys, Exp);
    M_all(k,:) = muz;
end
save(fullfile(outdir, 'example_squid_basic.mat'), 'S_vals', 'B_arr', 'M_all', 'T_K');
fprintf('  Saved: example_squid_basic.mat\n');
clear Sys Exp S_vals B_arr M_all k muz T_K;

%% ---- mu_eff(T): effective moment vs temperature ----
T_arr = linspace(2, 300, 50);
Exp.Field = 10;  % mT
S_vals = [0.5, 1.0, 1.5];
chi_all = zeros(length(S_vals), length(T_arr));
for k = 1:length(S_vals)
    Sys.S = S_vals(k); Sys.g = 2.0;
    Exp.Temperature = T_arr;
    [~, chi] = curry(Sys, Exp);
    chi_all(k,:) = chi;
end
save(fullfile(outdir, 'example_magnetization_tempdep.mat'), 'S_vals', 'T_arr', 'chi_all');
fprintf('  Saved: example_magnetization_tempdep.mat\n');
clear Sys Exp S_vals T_arr chi_all k chi;

fprintf('Done. All magnetometry reference files written to:\n  %s\n', outdir);
