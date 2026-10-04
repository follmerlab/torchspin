% generate_chili_refs.m
%
% Generate MATLAB chili() reference spectra for torchspin validation.
%
% Run from the repo root:
%   cd /path/to/EasySpin
%   matlab -batch "addpath('easyspin'); run('tests/generate_chili_refs.m')"
%
% Outputs (in tests/data/):
%   chili_ref_isog_notcorr.mat      – slightly anisotropic g, no nucleus, tcorr=1e-9 s
%   chili_ref_anisog.mat            – anisotropic g, no nucleus, tcorr=5e-10 s
%   chili_ref_14N_isoA.mat          – 14N, isotropic A, g anisotropy, tcorr=1e-9 s
%   chili_ref_1H_fastmotion.mat     – 1H, isotropic A, intermediate motion, tcorr=5e-9 s
%
% Note: all spectra use Harmonic=0 (absorption) to avoid derivative sign ambiguity.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'easyspin'));
out_dir = fullfile(fileparts(mfilename('fullpath')), 'data');

mwFreq  = 9.5;   % GHz
nPoints = 512;
Opt.LLMK      = [14 7 2 6];
Opt.Verbosity = 0;

%% -----------------------------------------------------------------------
%  Case 1: Isotropic g, no nucleus, tcorr = 1e-9 s
%  Pure diffusion broadening. torchspin should match well.
% -----------------------------------------------------------------------
Sys1.S     = 1/2;
Sys1.g     = [2.009 2.006 2.002];   % slightly anisotropic (MATLAB rejects isotropic)
Sys1.tcorr = 1e-9;
Sys1.lw    = 0;

Exp1.mwFreq  = mwFreq;
Exp1.Range   = [334 342];
Exp1.nPoints = nPoints;
Exp1.Harmonic = 0;

[B1, spc1] = chili(Sys1, Exp1, Opt);

save(fullfile(out_dir, 'chili_ref_isog_notcorr.mat'), ...
    'B1', 'spc1', 'mwFreq');
fprintf('Case 1 (iso g, no nuc): %d points, peak=%.4g\n', numel(B1), max(spc1));

%% -----------------------------------------------------------------------
%  Case 2: Anisotropic g, no nucleus, tcorr = 5e-10 s
%  Intermediate motion regime — tests rank-2 spatial-spin coupling.
% -----------------------------------------------------------------------
Sys2.S     = 1/2;
Sys2.g     = [2.008 2.006 2.003];
Sys2.tcorr = 5e-10;
Sys2.lw    = 0;

Exp2.mwFreq  = mwFreq;
Exp2.Range   = [333 343];
Exp2.nPoints = nPoints;
Exp2.Harmonic = 0;

[B2, spc2] = chili(Sys2, Exp2, Opt);

save(fullfile(out_dir, 'chili_ref_anisog.mat'), ...
    'B2', 'spc2', 'mwFreq');
fprintf('Case 2 (aniso g, no nuc): %d points, peak=%.4g\n', numel(B2), max(spc2));

%% -----------------------------------------------------------------------
%  Case 3: 14N nitroxide, isotropic A, anisotropic g, tcorr = 1e-9 s
%  Slow motion: 3-line pattern with motional broadening.
% -----------------------------------------------------------------------
Sys3.S     = 1/2;
Sys3.g     = [2.009 2.006 2.002];
Sys3.Nucs  = '14N';
Sys3.A     = [15 15 15];   % MHz  (isotropic A)
Sys3.tcorr = 1e-9;
Sys3.lw    = 0;

Exp3.mwFreq  = mwFreq;
Exp3.Range   = [328 348];
Exp3.nPoints = nPoints;
Exp3.Harmonic = 0;

[B3, spc3] = chili(Sys3, Exp3, Opt);

save(fullfile(out_dir, 'chili_ref_14N_isoA.mat'), ...
    'B3', 'spc3', 'mwFreq');
fprintf('Case 3 (14N isoA, tcorr=1e-9): %d points, peak=%.4g\n', numel(B3), max(spc3));

%% -----------------------------------------------------------------------
%  Case 4: 1H intermediate motion, tcorr = 5e-9 s
%  Intermediate-motion regime with 1H hyperfine — tests nuclear mI coupling.
% -----------------------------------------------------------------------
Sys4.S     = 1/2;
Sys4.g     = [2.008 2.006 2.003];
Sys4.Nucs  = '1H';
Sys4.A     = [10 10 10];   % MHz isotropic
Sys4.tcorr = 5e-9;
Sys4.lw    = [0 0.1];      % small Lorentzian broadening (mT)

Exp4.mwFreq  = mwFreq;
Exp4.Range   = [335 341];
Exp4.nPoints = nPoints;
Exp4.Harmonic = 0;

[B4, spc4] = chili(Sys4, Exp4, Opt);

save(fullfile(out_dir, 'chili_ref_1H_fastmotion.mat'), ...
    'B4', 'spc4', 'mwFreq');
fprintf('Case 4 (1H fast motion): %d points, peak=%.4g\n', numel(B4), max(spc4));

fprintf('\nAll chili reference files written to %s\n', out_dir);
