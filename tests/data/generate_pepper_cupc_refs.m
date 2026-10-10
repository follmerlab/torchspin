% EasySpin references for the Cu(II) phthalocyanine family that exposed the
% pepper defects reported against torchspin 0.3.0: a large axial copper
% hyperfine (A_par ~ 647 MHz) with up to four equivalent nitrogen ligands.
%
% Covers the three resonance solvers side by side on the same systems so that
% torchspin can be checked against EasySpin for each of them:
%   Opt.Method = 'matrix'  exact diagonalization
%   Opt.Method = 'perturb' second-order perturbation theory
%   Opt.Method = 'hybrid'  exact core + perturbational ligand nuclei
%                          (Opt.HybridCoreNuclei selects the exact nuclei)
%
% Both natural-abundance ('Cu','N') and single-isotope ('63Cu','14N') forms are
% stored, because the isotopologue expansion is itself one of the things under
% test.  Grids are converged ([91 4]); the EasySpin default [19 4] is also
% stored for two cases so the grid-convergence claim can be checked in both
% codes.
%
% Run from tests/data with EasySpin on the path:
%   matlab -batch "addpath('<...>/EasySpin/easyspin'); cd('<...>/tests/data'); generate_pepper_cupc_refs"
%
% Saved to ref_pepper_cupc.mat (-v7) as a cell array `cases`; each case stores
% name, Sys, Exp, Opt and the spectrum (B, spc).
clear all
fprintf('Generating pepper CuPc references...\n');
cases = {};

% Parameters from the MATLAB fit script Pc_Fits.m for
% 2020_02_20_CuPc_1_100_RR_AHF: axial Cu(II), four effectively isotropic N.
g_CuPc = [2.04894 2.181];        % [g_perp g_par]
A_Cu   = [15.3311 646.629];      % [A_perp A_par], MHz
A_N    = [45 45];                % MHz
lw_mT  = 0.542209;               % Gaussian FWHM

Exp = struct('mwFreq', 9.347144, 'Range', [233.8 433.8], 'nPoints', 2667, 'Harmonic', 1);

% ---------------------------------------------------------------------------
% Cu with 0..4 nitrogens, each solver, converged grid
% ---------------------------------------------------------------------------
% nN = 4 is skipped for 'matrix': EasySpin needs about 10 min for the 648-dim
% Hilbert space, and the point of 'hybrid' is that it does not.
iso = {'natural', {'Cu','N'}; 'isotope', {'63Cu','14N'}};
for iv = 1:size(iso,1)
  tag  = iso{iv,1};
  nucs = iso{iv,2};
  for nN = 0:4
    Sys = struct('S', 1/2, 'g', g_CuPc, 'lw', lw_mT);
    if nN > 0
      nuclist = nucs{1};
      for k = 1:nN, nuclist = [nuclist ',' nucs{2}]; end  %#ok<AGROW>
      Sys.Nucs = nuclist;
      Sys.A = [A_Cu; repmat(A_N, nN, 1)];
    end

    if nN <= 2
      cases{end+1} = mk(sprintf('cupc_%s_%dN_matrix', tag, nN), Sys, Exp, ...
                        struct('GridSize',[91 4],'Method','matrix'));           %#ok<AGROW>
    end
    cases{end+1} = mk(sprintf('cupc_%s_%dN_perturb', tag, nN), Sys, Exp, ...
                      struct('GridSize',[91 4],'Method','perturb'));            %#ok<AGROW>
    if nN > 0
      % Exact electron + Cu core, nitrogens by perturbation.
      cases{end+1} = mk(sprintf('cupc_%s_%dN_hybrid', tag, nN), Sys, Exp, ...
                        struct('GridSize',[91 4],'Method','hybrid', ...
                               'HybridCoreNuclei',1));                          %#ok<AGROW>
    end
  end
end

% ---------------------------------------------------------------------------
% Grid convergence: EasySpin's own default versus a converged grid
% ---------------------------------------------------------------------------
Sys = struct('S',1/2,'g',g_CuPc,'lw',lw_mT);
cases{end+1} = mk('cupc_grid19_0N_matrix', Sys, Exp, struct('GridSize',[19 4],'Method','matrix'));
Sys.Nucs = 'Cu,N,N,N,N';
Sys.A = [A_Cu; repmat(A_N,4,1)];
cases{end+1} = mk('cupc_grid19_4N_perturb', Sys, Exp, struct('GridSize',[19 4],'Method','perturb'));
cases{end+1} = mk('cupc_grid19_4N_hybrid', Sys, Exp, ...
                  struct('GridSize',[19 4],'Method','hybrid','HybridCoreNuclei',1));

% ---------------------------------------------------------------------------
% Hybrid with a larger exact core, and with an anisotropic / tilted ligand
% ---------------------------------------------------------------------------
% Core = electron + Cu + the first N, remaining N perturbational.
Sys = struct('S',1/2,'g',g_CuPc,'Nucs','Cu,N,N','A',[A_Cu; A_N; A_N],'lw',lw_mT);
cases{end+1} = mk('cupc_hybrid_core_CuN', Sys, Exp, ...
                  struct('GridSize',[91 4],'Method','hybrid','HybridCoreNuclei',[1 2]));

% Anisotropic nitrogen with a tilted A frame: exercises the full A tensor
% contraction in the nuclear sub-Hamiltonian, not just the isotropic case.
Sys = struct('S',1/2,'g',g_CuPc,'Nucs','63Cu,14N,14N', ...
             'A',[A_Cu; 40 55; 40 55], ...
             'AFrame',[0 0 0; 0 -pi/2 0; -pi/2 -pi/2 0], 'lw',lw_mT);
cases{end+1} = mk('cupc_hybrid_aframe', Sys, Exp, ...
                  struct('GridSize',[91 4],'Method','hybrid','HybridCoreNuclei',1));
cases{end+1} = mk('cupc_matrix_aframe', Sys, Exp, ...
                  struct('GridSize',[91 4],'Method','matrix'));

% Quadrupolar nitrogen: the sub-Hamiltonian picks up H_Q and the nuclear Zeeman
% term, which are independent of the electronic manifold.
% Q as three explicit principal values (traceless): a two-column Sys.Q is
% [eeqQ/h eta] in EasySpin, not axial shorthand, so three values keep the
% tensor unambiguous between the two codes.
Sys = struct('S',1/2,'g',g_CuPc,'Nucs','63Cu,14N','A',[A_Cu; A_N], ...
             'Q',[0 0 0; -0.6 -0.4 1.0],'lw',lw_mT);
cases{end+1} = mk('cupc_hybrid_quad', Sys, Exp, ...
                  struct('GridSize',[91 4],'Method','hybrid','HybridCoreNuclei',1));
cases{end+1} = mk('cupc_matrix_quad', Sys, Exp, ...
                  struct('GridSize',[91 4],'Method','matrix'));

% ---------------------------------------------------------------------------
% The full two-component fit model from Pc_Fits.m (Cu(II) + radical impurity)
% ---------------------------------------------------------------------------
Sys0 = struct('g',g_CuPc,'Nucs','Cu,N,N,N,N','A',[A_Cu; repmat(A_N,4,1)], ...
              'lw',lw_mT,'weight',1);
Sys1 = struct('g',2.003,'lw',0.01,'weight',0.001);
for m = {'perturb','hybrid'}
  Opt = struct('GridSize',[91 4],'Method',m{1},'Verbosity',0);
  if strcmp(m{1},'hybrid'), Opt.HybridCoreNuclei = 1; end
  [B,spc] = pepper({Sys0,Sys1}, Exp, Opt);
  cases{end+1} = struct('name',sprintf('cupc_twocomp_%s',m{1}), ...
                        'Sys',Sys0,'Sys2',Sys1,'Exp',Exp,'Opt',Opt, ...
                        'B',B,'spc',spc);                                       %#ok<AGROW>
end

script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_cupc.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));

function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0;
  fprintf('  %s\n', name);
  t0 = tic;
  [B,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc, ...
             'matlab_seconds',toc(t0));
end
