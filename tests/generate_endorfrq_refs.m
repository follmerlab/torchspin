% generate_endorfrq_refs.m
%
% Generate reference data for torchspin ENDOR (endorfrq + salt) validation.
%
% Run from the repo root:
%   cd /path/to/EasySpin
%   matlab -batch "addpath('easyspin'); run('tests/generate_endorfrq_refs.m')"
%
% Outputs:
%   tests/data/endorfrq_ref_1H_axial.mat     – S=1/2, 1H isotropic A, z-axis
%   tests/data/endorfrq_ref_14N_offaxis.mat  – S=1/2, 14N isotropic A, off-axis
%   tests/data/endorfrq_ref_1H_aniso.mat     – S=1/2, 1H anisotropic A, tilted
%   tests/data/salt_ref_1H.mat               – salt powder ENDOR spectrum, 1H
%   tests/data/salt_ref_14N.mat              – salt powder ENDOR spectrum, 14N
%
% Note: Exp.SampleFrame = [alpha, beta, gamma] (radians, z-y'-z'' passive).
%   For [0, beta, 0], the field is at theta=beta, phi=0 in the molecular frame,
%   matching torchspin endorfrq(sys, B, phi=0, theta=beta).

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'easyspin'));

out_dir = fullfile(fileparts(mfilename('fullpath')), 'data');

% Reference resonance field: free-electron at 9.5 GHz
mwFreq = 9.5;   % GHz
B_res  = 338.4; % mT (slightly below free-electron to be on EPR resonance with g=2.0)

%% -----------------------------------------------------------------------
%  Case 1: S=1/2 + 1H, isotropic A=10 MHz, field along z (theta=0)
% -----------------------------------------------------------------------
Sys1.S    = 1/2;
Sys1.g    = 2.0;
Sys1.Nucs = '1H';
Sys1.A    = [10 10 10];   % MHz

Exp1.mwFreq     = mwFreq;
Exp1.Field      = B_res;
Exp1.SampleFrame = [0 0 0];  % theta=0 -> field along z

[pos1, int1, tra1] = endorfrq(Sys1, Exp1);

save(fullfile(out_dir, 'endorfrq_ref_1H_axial.mat'), ...
    'pos1', 'int1', 'tra1', 'mwFreq', 'B_res');
fprintf('Case 1 (1H axial): %d lines\n', numel(pos1));

%% -----------------------------------------------------------------------
%  Case 2: S=1/2 + 14N, isotropic A=30 MHz, field at theta=pi/3
% -----------------------------------------------------------------------
Sys2.S    = 1/2;
Sys2.g    = 2.0;
Sys2.Nucs = '14N';
Sys2.A    = [30 30 30];   % MHz

Exp2.mwFreq     = mwFreq;
Exp2.Field      = B_res;
Exp2.SampleFrame = [0 pi/3 0];  % theta=pi/3, phi=0

[pos2, int2, tra2] = endorfrq(Sys2, Exp2);

save(fullfile(out_dir, 'endorfrq_ref_14N_offaxis.mat'), ...
    'pos2', 'int2', 'tra2', 'mwFreq', 'B_res');
fprintf('Case 2 (14N off-axis): %d lines\n', numel(pos2));

%% -----------------------------------------------------------------------
%  Case 3: S=1/2 + 1H, anisotropic A=[5 5 20] MHz, field at theta=pi/4
% -----------------------------------------------------------------------
Sys3.S    = 1/2;
Sys3.g    = 2.0;
Sys3.Nucs = '1H';
Sys3.A    = [5 5 20];     % MHz  (axial hyperfine)

Exp3.mwFreq     = mwFreq;
Exp3.Field      = B_res;
Exp3.SampleFrame = [0 pi/4 0];  % theta=pi/4, phi=0

[pos3, int3, tra3] = endorfrq(Sys3, Exp3);

save(fullfile(out_dir, 'endorfrq_ref_1H_aniso.mat'), ...
    'pos3', 'int3', 'tra3', 'mwFreq', 'B_res');
fprintf('Case 3 (1H anisotropic, tilted): %d lines\n', numel(pos3));

%% -----------------------------------------------------------------------
%  Case 4: salt powder ENDOR — S=1/2 + 1H, isotropic A=10 MHz
%  lw = 0.5 MHz (Gaussian) so spectra are broadened for cosine comparison
% -----------------------------------------------------------------------
SysS1.S       = 1/2;
SysS1.g       = 2.0;
SysS1.Nucs    = '1H';
SysS1.A       = [10 10 10];    % MHz
SysS1.lwEndor = 0.5;           % ENDOR linewidth [MHz] Gaussian FWHM

ExpS1.mwFreq  = mwFreq;
ExpS1.Field   = B_res;
ExpS1.Range   = [10 20];  % MHz (covers 14.0 ± 5.0 MHz)

OptS1.GridSize = 31;

[x_salt1, y_salt1] = salt(SysS1, ExpS1, OptS1);

save(fullfile(out_dir, 'salt_ref_1H.mat'), ...
    'x_salt1', 'y_salt1', 'mwFreq', 'B_res');
fprintf('Salt Case 1 (1H powder): %d points\n', numel(x_salt1));

%% -----------------------------------------------------------------------
%  Case 5: salt powder ENDOR — S=1/2 + 14N, isotropic A=30 MHz
%  lw = 0.5 MHz (Gaussian)
% -----------------------------------------------------------------------
SysS2.S       = 1/2;
SysS2.g       = 2.0;
SysS2.Nucs    = '14N';
SysS2.A       = [30 30 30];    % MHz
SysS2.lwEndor = 0.5;           % ENDOR linewidth [MHz] Gaussian FWHM

ExpS2.mwFreq  = mwFreq;
ExpS2.Field   = B_res;
ExpS2.Range   = [0 50];   % MHz

OptS2.GridSize = 31;

[x_salt2, y_salt2] = salt(SysS2, ExpS2, OptS2);

save(fullfile(out_dir, 'salt_ref_14N.mat'), ...
    'x_salt2', 'y_salt2', 'mwFreq', 'B_res');
fprintf('Salt Case 2 (14N powder): %d points\n', numel(x_salt2));

fprintf('\nAll reference files written to %s\n', out_dir);
