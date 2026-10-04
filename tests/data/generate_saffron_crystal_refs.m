% Generate saffron single-crystal cross-validation references.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
% Each file contains struct `data` with data.x, data.y (and y2 where noted).

fprintf('Generating saffron crystal reference files...\n\n');

% =========================================================================
% Case 1: 3pESEEM, single crystal, one and two sample orientations
% (from EasySpin tests/saffron_crystalorientations.m)
% =========================================================================
fprintf('Case 1: crystal orientations, 3pESEEM...\n');
clear Sys Exp Opt data
Sys.S = 1/2;
Sys.Nucs = '1H';
Sys.A_ = [5 2];               % aiso=5, aaniso=2 → [3 3 9]... (EasySpin conv)

Exp.Sequence = '3pESEEM';
Exp.dt = 0.01;
Exp.tau = 0.1;
Exp.nPoints = 120;
Exp.Field = 350;
Exp.MolFrame = [0 0 0];
Exp.SampleFrame = [0 -pi/2 0];

Opt.GridSize = 20;
Opt.Verbosity = 0;

[x1, y1] = saffron(Sys, Exp, Opt);

Exp.SampleFrame = [0 -pi/2 0; 0 0 0];
[x2, y2] = saffron(Sys, Exp, Opt);

data.x = x1(:)'; data.y = real(y1(:))';
data.x2 = x2(:)'; data.y2 = real(y2(:))';
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('saffron_crystalorientations.mat', 'data', '-v7');
fprintf('  saved saffron_crystalorientations.mat (max|y|=%.4g)\n\n', max(abs(data.y)));

% =========================================================================
% Case 2: HYSCORE, P212121 crystal (from EasySpin tests/saffron_crystal.m)
% =========================================================================
fprintf('Case 2: P212121 crystal HYSCORE...\n');
clear Sys Exp Opt data
Sys.Nucs = '1H';
Sys.A = [5 20];
Sys.lwEndor = 0.5;

Exp.MolFrame = [pi/3 pi/6 pi/4];
Exp.SampleFrame = [pi/9 pi/5 0];
Exp.CrystalSymmetry = 'P212121';
Exp.Sequence = 'HYSCORE';
Exp.Field = 330;
Exp.tau = 0.080;
Exp.dt = 0.120;
Exp.nPoints = 256;

Opt.Verbosity = 0;
[x, y] = saffron(Sys, Exp, Opt);

data.y = real(y);
data.Sys = Sys; data.Exp = Exp; data.Opt = Opt;
save('saffron_crystal_hyscore.mat', 'data', '-v7');
fprintf('  saved saffron_crystal_hyscore.mat (max|y|=%.4g)\n\n', max(abs(data.y(:))));

fprintf('All saffron crystal reference files generated.\n');
