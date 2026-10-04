% Generate reference for two-component MimsENDOR (saffron_twocomponents.m).
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.

Exp.Sequence = 'MimsENDOR';
Exp.Field = 325;    % mT
Exp.tau = 0.1;      % µs
Exp.Range = larmorfrq('1H',Exp.Field) + [-1 1]*10;  % MHz

Sys1.Nucs = '1H';
Sys1.A = [1 3];     % MHz (axial: [1 1 3])
Sys1.lwEndor = 0.1;

Sys2.Nucs = '1H';
Sys2.A = [7 9];     % MHz (axial: [7 7 9])
Sys2.lwEndor = 0.1;
Sys2.weight = 0.3;

Opt.Verbosity = 0;
[x, y] = saffron({Sys1,Sys2}, Exp, Opt);
y = y/max(abs(y));

data.x = x(:)'; data.y = y(:)';
save('saffron_twocomponents.mat', 'data', '-v7');
fprintf('saved saffron_twocomponents.mat (%d pts)\n', numel(y));
