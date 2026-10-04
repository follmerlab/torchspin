% Generate curry (magnetometry) and blochsteady cross-validation references.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.

fprintf('Generating curry/blochsteady reference files...\n\n');

% =========================================================================
% Case 1: magnetization of AF-coupled dimer (curry_simplemag)
% =========================================================================
fprintf('Case 1: curry simplemag...\n');
clear Sys Exp Opt data
Sys.S = [1/2 1/2];
Sys.g = [2 2];
Sys.ee = -2*-4*30e3;   % MHz

Exp.Temperature = 1;               % K
Exp.Field = linspace(0,17,20)*1e3; % mT

Opt.Output = 'muBM';
m = curry(Sys,Exp,Opt);

data.B = Exp.Field; data.T = Exp.Temperature; data.muBM = m(:)';
save('curry_simplemag.mat','data','-v7');
fprintf('  saved curry_simplemag.mat\n\n');

% =========================================================================
% Case 2: susceptibility of the same dimer vs T (curry_simplechi)
% =========================================================================
fprintf('Case 2: curry simplechi...\n');
clear Exp Opt data
Exp.Temperature = 1:100;  % K
Exp.Field = 1000;         % mT

Opt.Output = 'chimol';    % SI, m^3/mol
chi_SI = curry(Sys,Exp,Opt);

data.B = Exp.Field; data.T = Exp.Temperature; data.chimol = chi_SI(:)';
save('curry_simplechi.mat','data','-v7');
fprintf('  saved curry_simplechi.mat\n\n');

% =========================================================================
% Case 3: Gd(III)-Cu(II) mueff + chimol vs T (curry_mueffGdCu, B=100 mT)
% =========================================================================
fprintf('Case 3: curry GdCu...\n');
clear Sys Exp Opt data
Sys.S = [7/2 1/2];
Sys.ee = -2*5*30e3;   % MHz

Exp.Temperature = 0.5:0.5:125;  % K (avoid T=0)
Exp.Field = 100;                % mT

Opt.Output = 'chimol mueff';
Opt.deltaB = 0.01;              % mT
[chi_SI, mueff] = curry(Sys,Exp,Opt);

data.B = Exp.Field; data.T = Exp.Temperature;
data.chimol = chi_SI(:)'; data.mueff = mueff(:)';
save('curry_GdCu.mat','data','-v7');
fprintf('  saved curry_GdCu.mat\n\n');

% =========================================================================
% Case 4: S=1 with zero-field splitting, chimol vs T
% =========================================================================
fprintf('Case 4: curry S=1 + D...\n');
clear Sys Exp Opt data
Sys.S = 1;
Sys.g = 2.1;
Sys.D = 5e3;   % MHz (5 GHz)

Exp.Temperature = 1:100;  % K
Exp.Field = 100;          % mT

Opt.Output = 'chimol';
chi_SI = curry(Sys,Exp,Opt);

data.B = Exp.Field; data.T = Exp.Temperature; data.chimol = chi_SI(:)';
save('curry_S1_D.mat','data','-v7');
fprintf('  saved curry_S1_D.mat\n\n');

% =========================================================================
% Case 5: blochsteady_simple
% =========================================================================
fprintf('Case 5: blochsteady simple...\n');
clear data
g = gfree; T1 = 20; T2 = 1;
deltaB0 = 0.1; B1 = 0.02; ModAmp = 0.5; ModFreq = 50;

[t, Mx, My, Mz] = blochsteady(g,T1,T2,deltaB0,B1,ModAmp,ModFreq);

data.t = t(:)'; data.Mx = Mx(:)'; data.My = My(:)'; data.Mz = Mz(:)';
data.params = [g T1 T2 deltaB0 B1 ModAmp ModFreq];
save('blochsteady_simple.mat','data','-v7');
fprintf('  saved blochsteady_simple.mat (%d pts)\n\n', numel(t));

fprintf('All curry/blochsteady reference files generated.\n');
