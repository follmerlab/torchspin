% EasySpin references for pepper Opt.Method='hybrid', whose exact core is built
% with nucspinrmv.  The cores carry more than the bare electron spin: an
% exchange-coupled dimer (Sys.ee) and an S = 3/2 system with a higher-order
% Zeeman term (Sys.Ham132).  (saffron, the other user of nucspinrmv in
% torchspin, is not covered: EasySpin's saffron rejects Sys.Ham*.)  EasySpin's nucspinrmv keeps every field of Sys;
% torchspin <= 0.8.1 dropped ee and Ham there (bug B14).
%
% Each case is simulated with Opt.Method = 'hybrid' and 'matrix'.
%
% Run from tests/data with EasySpin on the path:
%   matlab -batch "addpath(genpath('<...>/easyspin')); cd('<...>/tests/data'); generate_nucspin_core_refs"
%
% Saved to ref_nucspin_core.mat (-v7): a struct per case with fields
% B (mT), hybrid, matrix.
clear all
Exp = struct('mwFreq', 9.5, 'Range', [100 600], 'nPoints', 2000);
Opt0 = struct('GridSize', [31 4]);

% 1) Anisotropic exchange-coupled dimer, Cu hyperfine exact, 1H perturbational
S1 = struct('S', [1/2 1/2], 'g', [2 2 2.05; 2.1 2.1 2.0], 'ee', [2900 2900 3200], ...
            'Nucs', '63Cu,1H', 'A', [50 50 500 0 0 0; 3 3 3 0 0 0], 'lw', [1 0]);
% 2) Isotropic dimer, J = 1000 MHz, one 1H coupled to both electrons, no core nuclei
S2 = struct('S', [1/2 1/2], 'g', [2.0; 2.1], 'ee', 1000, ...
            'Nucs', '1H', 'A', [4 4 4 2 2 2], 'lw', [1 0]);
% 3) S = 3/2 with zero-field splitting and an axial Ham132 term, 1H perturbational
S3 = struct('S', 3/2, 'g', [2 2 2], 'D', [300 30], 'Ham132', [0 0 0.05 0 0], ...
            'Nucs', '1H', 'A', [6 6 6], 'lw', [1 0]);
systems = {S1, S2, S3};
names = {'dimer_aniso_cu_core', 'dimer_iso', 'S32_ham132'};
cores = {1, [], []};

for k = 1:numel(systems)
  Opt = Opt0;
  [B, spc_m] = pepper(systems{k}, Exp, Opt);
  Opt.Method = 'hybrid';
  Opt.HybridCoreNuclei = cores{k};
  [~, spc_h] = pepper(systems{k}, Exp, Opt);
  refs.(names{k}) = struct('B', B, 'matrix', spc_m, 'hybrid', spc_h);
  c = (spc_h(:)'*spc_m(:))/norm(spc_h)/norm(spc_m);
  fprintf('%s: cos(hybrid, matrix) = %.6f\n', names{k}, c);
end

save('ref_nucspin_core.mat', '-struct', 'refs', '-v7');
