% Generate a small (32x32) reference for the user-defined HYSCORE sequence.
% Same physics as EasySpin tests/saffron_customsequence.m, reduced nPoints
% so the Python cross-validation test runs in seconds.
%
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.

Sys.Nucs = '14N';
Sys.A = [-1 -1 2]*0.5+0.8;
Sys.Q = 1;

Exp.Field = 330;
tau = 0.080;  % µs
t0 = 0;       % µs
p90.Flip = pi/2;
p180.Flip = pi;
Exp.Sequence = {p90 tau p90 t0 p180 t0 p90 tau};
Exp.nPoints = [32 32];
Exp.Dim1 = {'d2' 0.1};
Exp.Dim2 = {'d3' 0.1};

Opt.Verbosity = 0;
y = saffron(Sys, Exp, Opt);

data.y = real(y);
data.Sys = Sys; data.Exp = Exp;
save('saffron_customsequence_small.mat', 'data', '-v7');
fprintf('saved saffron_customsequence_small.mat, max|y|=%g\n', max(abs(data.y(:))));
