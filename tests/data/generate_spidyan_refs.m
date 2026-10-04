% Generate spidyan cross-validation references.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
% Each .mat holds struct `data` with data.t, data.sig_re, data.sig_im
% (signal transposed to row-major time axis where applicable).
% Cases adapted from EasySpin's tests/spidyan_*.m.

fprintf('Generating spidyan reference files...\n\n');
orig_state = warning; warning('off','all');

% =========================================================================
% Case 1: S=1/2, chirp pulse pair, two detection operators (spinonehalf)
% =========================================================================
fprintf('Case 1: spinonehalf...\n');
clear Sys Exp Opt Pulse data
Sys.S = 1/2;
Sys.ZeemanFreq = 33.500;

Pulse.Type = 'quartersin/linear';
Pulse.trise = 0.015;
Pulse.tp = 0.1;
Pulse.Flip = pi;
Pulse.Frequency = 1000*[-0.100 0.100];

Exp.Sequence = {Pulse 0.5 Pulse};
Exp.Field = 1240;
Exp.mwFreq = 33.5;
Exp.DetSequence = [1 1 1];
Exp.DetOperator = {'z1','+1'};
Exp.DetFreq = [0 33.5];

Opt.IntTimeStep = 0.0001;
Opt.SimFreq = 32;

[t, signal] = spidyan(Sys,Exp,Opt);
data.t = t(:)'; data.sig_re = real(signal).'; data.sig_im = imag(signal).';
save('spidyan_spinonehalf.mat','data','-v7');
fprintf('  saved spidyan_spinonehalf.mat (%d pts)\n\n', numel(t));

% =========================================================================
% Case 2: electron-nucleus (1H, A=[8 45 45]) (electronnucleus)
% =========================================================================
fprintf('Case 2: electronnucleus...\n');
clear Sys Exp Opt Pulse data
Sys.S = 1/2;
Sys.g = gfree;
Sys = nucspinadd(Sys,'1H',[8 45 45]);

Pulse.Type = 'quartersin/linear';
Pulse.trise = 0.015;
Pulse.tp = 0.1;
Pulse.Frequency = 1000*[-0.100 0.100];
Pulse.Flip = pi;

Exp.Sequence = {Pulse 0.5};
Exp.Field = 1195;
Exp.mwFreq = 33.5;
Exp.DetSequence = [1 1];
Exp.DetPhase = 0;
Exp.DetOperator = {'z1'};
Exp.DetFreq = 0;

Opt.IntTimeStep = 0.0001;
Opt.SimFreq = 32;

[t, signal] = spidyan(Sys,Exp,Opt);
data.t = t(:)'; data.sig_re = real(signal(:))'; data.sig_im = imag(signal(:))';
save('spidyan_electronnucleus.mat','data','-v7');
fprintf('  saved spidyan_electronnucleus.mat (%d pts)\n\n', numel(t));

% =========================================================================
% Case 3: high spin S=3/2 with D (highspin)
% =========================================================================
fprintf('Case 3: highspin...\n');
clear Sys Exp Opt Pulse data
Sys.S = 3/2;
Sys.D = 200;
Sys.g = gfree;

Pulse.Type = 'quartersin/linear';
Pulse.trise = 0.015;
Pulse.tp = 0.1;
Pulse.Frequency = 1000*[-0.100 0.100];
Pulse.Flip = pi;

Exp.Sequence = {Pulse 0.5};
Exp.Field = 1195;
Exp.mwFreq = 33.5;
Exp.DetSequence = [1 1];
Exp.DetPhase = 0;
Exp.DetOperator = {'z1'};

Opt.IntTimeStep = 0.0001;
Opt.SimFreq = 32;

[t, signal] = spidyan(Sys,Exp,Opt);
data.t = t(:)'; data.sig_re = real(signal(:))'; data.sig_im = imag(signal(:))';
save('spidyan_highspin.mat','data','-v7');
fprintf('  saved spidyan_highspin.mat (%d pts)\n\n', numel(t));

% =========================================================================
% Case 4: phase cycling (phasecycling)
% =========================================================================
fprintf('Case 4: phasecycling...\n');
clear Sys Exp Opt Pulse Pulse2 data
Sys.S = 1/2;
Sys.ZeemanFreq = 33.500;

Pulse.Type = 'rectangular';
Pulse.tp = 0.1;
Pulse.Flip = pi;
Pulse2 = Pulse;
Pulse2.Phase = pi;

PC = [0, 1; pi, -1];

Exp.Sequence = {Pulse Pulse2 Pulse};
Exp.Field = 1240;
Exp.mwFreq = 33.5;
Exp.DetSequence = 1;
Exp.PhaseCycle = {PC [] PC};
Exp.DetPhase = 0;
Exp.DetOperator = {'z1'};

Opt.IntTimeStep = 0.0001;
Opt.SimFreq = 32;

[t, signal] = spidyan(Sys,Exp,Opt);
data.t = t(:)'; data.sig_re = real(signal(:))'; data.sig_im = imag(signal(:))';
save('spidyan_phasecycling.mat','data','-v7');
fprintf('  saved spidyan_phasecycling.mat (%d pts)\n\n', numel(t));

% =========================================================================
% Case 5: 1D indirect dimension, varying flip angle (onedimensionalexperiment)
% =========================================================================
fprintf('Case 5: onedimensional...\n');
clear Sys Exp Opt Pulse data
Sys.S = 1/2;
Sys.ZeemanFreq = 33.500;

Pulse.Type = 'rectangular';
Pulse.tp = 0.1;
Pulse.Flip = pi;

Exp.Sequence = {Pulse 0.5 Pulse};
Exp.Field = 1240;
Exp.mwFreq = 33.5;
Exp.DetSequence = 1;
Exp.nPoints = 3;
Exp.Dim1 = {'p1.Flip' 0.05};
Exp.DetPhase = 0;
Exp.DetOperator = {'z1'};

Opt.IntTimeStep = 0.0001;
Opt.SimFreq = 32;

[t, signal] = spidyan(Sys,Exp,Opt);
if iscell(signal)
  sig = cell2mat(cellfun(@(s) s(:).', signal(:), 'UniformOutput', false));
  tax = t(1,:);
else
  sig = signal; tax = t;
end
data.t = tax(:)'; data.sig_re = real(sig); data.sig_im = imag(sig);
save('spidyan_onedimensional.mat','data','-v7');
fprintf('  saved spidyan_onedimensional.mat (size %dx%d)\n\n', size(sig,1), size(sig,2));

% =========================================================================
% Case 6: two coupled electrons (electronelectron)
% =========================================================================
fprintf('Case 6: electronelectron...\n');
clear Sys Exp Opt Pulse data
Sys.S = [1/2 1/2];
Sys.g = [gfree gfree];
Sys.J = 0;

Pulse.Type = 'quartersin/linear';
Pulse.trise = 0.015;
Pulse.tp = 0.1;
Pulse.Frequency = 1000*[-0.100 0.100];
Pulse.Flip = pi;

Exp.Sequence = {Pulse 0.5};
Exp.Field = 1195;
Exp.mwFreq = 33.5;
Exp.DetSequence = [1 1];
Exp.DetFreq = 0;
Exp.DetPhase = 0;

Opt.IntTimeStep = 0.0001;
Opt.SimFreq = 32;

[t, signal] = spidyan(Sys,Exp,Opt);
data.t = t(:)'; data.sig_re = real(signal).'; data.sig_im = imag(signal).';
save('spidyan_electronelectron.mat','data','-v7');
fprintf('  saved spidyan_electronelectron.mat (%d pts)\n\n', numel(t));

% =========================================================================
% Case 7: relaxation T1/T2 during chirp + delay (adapted from
% relaxationcomplexexcitation, standard excitation, with detection)
% =========================================================================
fprintf('Case 7: relaxation...\n');
clear Sys Exp Opt Pulse data
Sys.S = 1/2;
Sys.ZeemanFreq = 33.500;
Sys.T1 = 1;
Sys.T2 = 0.4;

Pulse.Type = 'quartersin/linear';
Pulse.trise = 0.015;
Pulse.tp = 0.1;
Pulse.Flip = pi;
Pulse.Frequency = 1000*[-0.100 0.100];

Exp.Sequence = {Pulse 0.5};
Exp.Field = 1240;
Exp.mwFreq = 33.5;
Exp.DetSequence = [1 1];
Exp.DetPhase = 0;
Exp.DetOperator = {'z1'};
Exp.DetFreq = 0;

Opt.IntTimeStep = 0.0001;
Opt.SimFreq = 32;
Opt.Relaxation = 1;

[t, signal] = spidyan(Sys,Exp,Opt);
data.t = t(:)'; data.sig_re = real(signal(:))'; data.sig_im = imag(signal(:))';
save('spidyan_relaxation.mat','data','-v7');
fprintf('  saved spidyan_relaxation.mat (%d pts)\n\n', numel(t));

warning(orig_state);
fprintf('All spidyan reference files generated.\n');
