% Generate saffron cross-validation references for S > 1/2 systems.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
%   >> cd tests/data
%   >> generate_saffron_highspin_refs
%
% Saves one .mat file per case.  Each file contains a struct `data` with:
%   data.x  – frequency or time axis returned by saffron
%   data.y  – signal (real part of time domain, or FD spectrum)
%   data.Sys, data.Exp, data.Opt  – inputs for reproducibility

fprintf('Generating saffron S>1/2 reference files...\n\n');

% =========================================================================
% Case 1: S=1 (triplet) + 1H, 2pESEEM
% =========================================================================
fprintf('Case 1: S=1 + 1H, 2pESEEM...\n');
Sys1.S   = 1;
Sys1.g   = [2.002 2.001 2.000];
Sys1.D   = [100 0];            % D=100 MHz, E=0
Sys1.Nucs = '1H';
Sys1.A   = [3 3 9];            % MHz, axial
Sys1.lwEndor = 0;

Exp1.Field    = 350;           % mT
Exp1.Sequence = '2pESEEM';
Exp1.dt       = 0.01;          % µs
Exp1.nPoints  = 256;
Exp1.tau      = 0.2;           % µs

Opt1.GridSize   = 30;
Opt1.Verbosity  = 0;
Opt1.SimulationMode = 'fast';

[x1, y1] = saffron(Sys1, Exp1, Opt1);

data.x = x1(:)';
data.y = real(y1(:)');
data.Sys = Sys1;
data.Exp = Exp1;
data.Opt = Opt1;
save('saffron_S1_2pESEEM.mat', 'data', '-v7');
fprintf('  saved saffron_S1_2pESEEM.mat  (nPoints=%d, max|y|=%.4g)\n\n', ...
    length(data.y), max(abs(data.y)));
clear data;

% =========================================================================
% Case 2: S=1 (triplet) + 1H, 3pESEEM
% =========================================================================
fprintf('Case 2: S=1 + 1H, 3pESEEM...\n');
Sys2 = Sys1;

Exp2.Field    = 350;
Exp2.Sequence = '3pESEEM';
Exp2.dt       = 0.01;
Exp2.nPoints  = 256;
Exp2.tau      = 0.2;

Opt2.GridSize   = 30;
Opt2.Verbosity  = 0;
Opt2.SimulationMode = 'fast';

[x2, y2] = saffron(Sys2, Exp2, Opt2);

data.x = x2(:)';
data.y = real(y2(:)');
data.Sys = Sys2;
data.Exp = Exp2;
data.Opt = Opt2;
save('saffron_S1_3pESEEM.mat', 'data', '-v7');
fprintf('  saved saffron_S1_3pESEEM.mat  (nPoints=%d, max|y|=%.4g)\n\n', ...
    length(data.y), max(abs(data.y)));
clear data;

% =========================================================================
% Case 3: S=3/2 + 14N, HYSCORE
% =========================================================================
fprintf('Case 3: S=3/2 + 14N, HYSCORE...\n');
Sys3.S    = 3/2;
Sys3.g    = [2.002 2.001 2.000];
Sys3.D    = [50 0];            % D=50 MHz, E=0
Sys3.Nucs = '14N';
Sys3.A    = [5 5 12];          % MHz
Sys3.Q    = [0.3 0 0];         % [Qxx Qyy Qzz] or [e2qQ/h, eta]

Exp3.Field    = 330;
Exp3.Sequence = 'HYSCORE';
Exp3.dt       = [0.02 0.02];   % µs
Exp3.nPoints  = [64 64];
Exp3.tau      = 0.2;

Opt3.GridSize   = 20;
Opt3.Verbosity  = 0;
Opt3.SimulationMode = 'fast';

[x3, y3] = saffron(Sys3, Exp3, Opt3);

data.x = x3;
data.y = real(y3);
data.Sys = Sys3;
data.Exp = Exp3;
data.Opt = Opt3;
save('saffron_S32_HYSCORE.mat', 'data', '-v7');
fprintf('  saved saffron_S32_HYSCORE.mat  (size=[%s], max|y|=%.4g)\n\n', ...
    num2str(size(data.y)), max(abs(data.y(:))));
clear data;

% =========================================================================
% Case 4: S=1 + 1H, 2pESEEM with orientation selection (mwFreq set)
% =========================================================================
fprintf('Case 4: S=1 + 1H, 2pESEEM with orientation selection...\n');
Sys4 = Sys1;

Exp4.Field       = 350;
Exp4.Sequence    = '2pESEEM';
Exp4.dt          = 0.01;
Exp4.nPoints     = 256;
Exp4.tau         = 0.2;
Exp4.mwFreq      = 9.5;        % GHz
Exp4.ExciteWidth = 100;        % MHz

Opt4.GridSize   = 30;
Opt4.Verbosity  = 0;
Opt4.SimulationMode = 'fast';

[x4, y4] = saffron(Sys4, Exp4, Opt4);

data.x = x4(:)';
data.y = real(y4(:)');
data.Sys = Sys4;
data.Exp = Exp4;
data.Opt = Opt4;
save('saffron_S1_2pESEEM_orisel.mat', 'data', '-v7');
fprintf('  saved saffron_S1_2pESEEM_orisel.mat  (nPoints=%d, max|y|=%.4g)\n\n', ...
    length(data.y), max(abs(data.y)));
clear data;

% =========================================================================
% Case 5: S=1 + 14N (quadrupole), 3pESEEM
% =========================================================================
fprintf('Case 5: S=1 + 14N (quadrupole), 3pESEEM...\n');
Sys5.S    = 1;
Sys5.g    = [2.002 2.001 2.000];
Sys5.D    = [100 0];
Sys5.Nucs = '14N';
Sys5.A    = [5 5 12];
Sys5.Q    = [0.3 0 0];

Exp5.Field    = 350;
Exp5.Sequence = '3pESEEM';
Exp5.dt       = 0.01;
Exp5.nPoints  = 256;
Exp5.tau      = 0.2;

Opt5.GridSize   = 30;
Opt5.Verbosity  = 0;
Opt5.SimulationMode = 'fast';

[x5, y5] = saffron(Sys5, Exp5, Opt5);

data.x = x5(:)';
data.y = real(y5(:)');
data.Sys = Sys5;
data.Exp = Exp5;
data.Opt = Opt5;
save('saffron_S1_14N_3pESEEM.mat', 'data', '-v7');
fprintf('  saved saffron_S1_14N_3pESEEM.mat  (nPoints=%d, max|y|=%.4g)\n\n', ...
    length(data.y), max(abs(data.y)));
clear data;

fprintf('Done. All 5 reference files written to current directory.\n');
