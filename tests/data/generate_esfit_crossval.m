% Generate cross-validation spectra for torchspin esfit tests.
% Run in MATLAB with EasySpin on the path.
% Saves .mat files that Python loads to fit with esfit+pepper.

% =========================================================================
% Case 1: S=1/2, isotropic g, Gaussian lw
% =========================================================================
fprintf('Generating esfit_crossval_case1.mat...\n');
Sys1.S   = 1/2;
Sys1.g   = 2.005;
Sys1.lw  = [0.8, 0.0];   % mT Gaussian FWHM

Exp1.mwFreq  = 9.5;        % GHz
Exp1.Range   = [330, 345]; % mT
Exp1.nPoints = 512;
Exp1.Harmonic = 1;

Opt1.GridSize   = 50;
Opt1.Verbosity  = 0;

[B1, spc1] = pepper(Sys1, Exp1, Opt1);

save('esfit_crossval_case1.mat', 'Sys1', 'Exp1', 'Opt1', 'B1', 'spc1', '-v7');
fprintf('  g=%.4f lw=%.2f mT  peak=%g\n', Sys1.g, Sys1.lw(1), max(abs(spc1)));

% =========================================================================
% Case 2: S=1/2, rhombic g [gx,gy,gz], Gaussian lw
% =========================================================================
fprintf('Generating esfit_crossval_case2.mat...\n');
Sys2.S   = 1/2;
Sys2.g   = [2.008, 2.005, 2.001];
Sys2.lw  = [1.0, 0.0];

Exp2.mwFreq  = 9.5;
Exp2.Range   = [330, 345];
Exp2.nPoints = 512;
Exp2.Harmonic = 1;

Opt2.GridSize   = 50;
Opt2.Verbosity  = 0;

[B2, spc2] = pepper(Sys2, Exp2, Opt2);

save('esfit_crossval_case2.mat', 'Sys2', 'Exp2', 'Opt2', 'B2', 'spc2', '-v7');
fprintf('  g=[%.4f %.4f %.4f] lw=%.2f mT\n', Sys2.g(1), Sys2.g(2), Sys2.g(3), Sys2.lw(1));

% =========================================================================
% Case 3: S=1/2 + 14N hyperfine, isotropic g and A, Gaussian lw
% =========================================================================
fprintf('Generating esfit_crossval_case3.mat...\n');
Sys3.S     = 1/2;
Sys3.g     = 2.006;
Sys3.Nucs  = '14N';
Sys3.A     = 42.0;         % MHz isotropic
Sys3.lw    = [0.5, 0.0];

Exp3.mwFreq  = 9.5;
Exp3.Range   = [330, 348];
Exp3.nPoints = 512;
Exp3.Harmonic = 1;

Opt3.GridSize   = 30;
Opt3.Verbosity  = 0;

[B3, spc3] = pepper(Sys3, Exp3, Opt3);

save('esfit_crossval_case3.mat', 'Sys3', 'Exp3', 'Opt3', 'B3', 'spc3', '-v7');
fprintf('  g=%.4f A=%.1f MHz lw=%.2f mT\n', Sys3.g, Sys3.A, Sys3.lw(1));

fprintf('\nDone. Files written to current directory.\n');
