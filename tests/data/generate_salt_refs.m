% Generate salt (powder ENDOR) cross-validation references.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
%   >> cd tests/data
%   >> generate_salt_refs
%
% Saves one .mat file per case.  Each file contains a struct `data` with:
%   data.x  – ENDOR frequency axis (MHz) returned by salt
%   data.y  – ENDOR spectrum (a.u.)
%   data.Sys, data.Exp, data.Opt  – inputs for reproducibility
%
% Cases are adapted from EasySpin's own test suite (tests/salt_*.m) to
% cover: rhombic g + axial 1H A, tilted AFrame, 14N with quadrupole,
% orientation selection via ExciteWidth, W-band gx/gz selection,
% two nuclei, axial-g orientation preselection, and S=3/2.

fprintf('Generating salt reference files...\n\n');

% =========================================================================
% Case 1: rhombic g + 1H axial A (from salt_invariance_axial1)
% =========================================================================
fprintf('Case 1: rhombic g + 1H, X-band...\n');
clear Sys Exp Opt data
Sys.S = 1/2;
Sys.g = [2 2.01 2.02];
Sys.Nucs = '1H';
Sys.A = 2*[-1 -1 2];           % MHz
Sys.lwEndor = 0.1;

Exp.Field  = 326.5;            % mT
Exp.Range  = [10 20];          % MHz (RF)
Exp.mwFreq = 9.7;              % GHz

Opt.GridSize  = [20 5];
Opt.Threshold = 1e-5;
Opt.Intensity = 'on';
Opt.Enhancement = 'off';
Opt.Verbosity = 0;

[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_1H_rhombicg.mat', 'data', '-v7');
fprintf('  saved salt_1H_rhombicg.mat (max y=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 2: same system, tilted AFrame (rotational invariance partner)
% =========================================================================
fprintf('Case 2: tilted AFrame...\n');
clear data
Sys.AFrame = pi/180*[23 72 -12];
[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_1H_aframe.mat', 'data', '-v7');
fprintf('  saved salt_1H_aframe.mat (max y=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 3: 14N + quadrupole, W-band, full excitation (from salt_excitewidth)
% =========================================================================
fprintf('Case 3: 14N + quadrupole, W-band...\n');
clear Sys Exp Opt data
Sys.S = 1/2;
Sys.g = [2.2 2.2 2];
Sys.Nucs = '14N';
Sys.A = [4 4 5];               % MHz
Sys.Q = -0.84*[-1 -1 2];       % MHz
Sys.HStrain = [1 1 1];
Sys.lwEndor = 0.1;

Exp.Field   = 3394;            % mT
Exp.Range   = [0 35];          % MHz
Exp.nPoints = 4096;
Exp.mwFreq  = 95;              % GHz

Opt.GridSize  = 50;
Opt.Threshold = 1e-4;
Opt.Intensity = 'on';
Opt.Verbosity = 0;

[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_14N_Wband.mat', 'data', '-v7');
fprintf('  saved salt_14N_Wband.mat (max y=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 4: 14N, W-band, orientation selection at gz (ExciteWidth)
% =========================================================================
fprintf('Case 4: 14N orientation-selective, gz...\n');
clear data
Exp.ExciteWidth = 3e3;         % MHz
[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_14N_orisel_gz.mat', 'data', '-v7');
fprintf('  saved salt_14N_orisel_gz.mat (max y=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 5: 14N, orientation selection at gx (mwFreq shifted)
% =========================================================================
fprintf('Case 5: 14N orientation-selective, gx...\n');
clear data
Exp.mwFreq = 104.51;           % GHz — resonant at gx for same field
[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_14N_orisel_gx.mat', 'data', '-v7');
fprintf('  saved salt_14N_orisel_gx.mat (max y=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 6: axial g + 1H, X-band orientation selection (from salt_preselectsimple)
% =========================================================================
fprintf('Case 6: axial g + 1H, X-band orisel...\n');
clear Sys Exp Opt data
Sys.S = 1/2;
Sys.g = [2 2 2.2];
Sys.Nucs = '1H';
Sys.A = 4 + [-1 -1 2]*1.5;     % MHz
Sys.lwEndor = 0.02;

Exp.Range = [8 18];            % MHz
Exp.mwFreq = 9.5;              % GHz
Exp.ExciteWidth = 100;         % MHz
% Field resonant at g = 2.1 (mid-spectrum single-crystal-like selection)
Exp.Field = Exp.mwFreq*1e9*planck/bmagn/2.1*1e3;   % mT

Opt.GridSize = 61;
Opt.Verbosity = 0;

[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_1H_orisel_axialg.mat', 'data', '-v7');
fprintf('  saved salt_1H_orisel_axialg.mat (max y=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 7: two nuclei (1H + 14N)
% =========================================================================
fprintf('Case 7: 1H + 14N, X-band...\n');
clear Sys Exp Opt data
Sys.S = 1/2;
Sys.g = [2.0023 2.0023 2.0023];
Sys.Nucs = '1H,14N';
Sys.A = [2 2 6; 1 1 3];        % MHz
Sys.Q = [0 0 0; -0.5 -0.5 1];  % MHz
Sys.lwEndor = 0.1;

Exp.Field  = 350;              % mT
Exp.Range  = [0 25];           % MHz
Exp.mwFreq = 9.8;              % GHz

Opt.GridSize = 31;
Opt.Verbosity = 0;

[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_1H14N_twonuc.mat', 'data', '-v7');
fprintf('  saved salt_1H14N_twonuc.mat (max y=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 8: S=3/2 + 1H (high-spin, from salt_pt_threehalf_onehalf pattern)
% =========================================================================
fprintf('Case 8: S=3/2 + 1H...\n');
clear Sys Exp Opt data
Sys.S = 3/2;
Sys.g = [2.0 2.0 2.0];
Sys.D = [500 0];               % MHz
Sys.Nucs = '1H';
Sys.A = [3 3 8];               % MHz
Sys.lwEndor = 0.1;

Exp.Field  = 350;              % mT
Exp.Range  = [5 25];           % MHz
Exp.mwFreq = 9.8;              % GHz

Opt.GridSize = 31;
Opt.Verbosity = 0;

[x, y] = salt(Sys, Exp, Opt);
data.x = x(:)'; data.y = y(:)';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('salt_S32_1H.mat', 'data', '-v7');
fprintf('  saved salt_S32_1H.mat (max y=%.4g)\n\n', max(abs(data.y)));

fprintf('All salt reference files generated.\n');
