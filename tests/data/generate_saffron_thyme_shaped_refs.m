% Generate MATLAB references for saffron's real-pulse route with *shaped*
% pulses -- EasySpin's own saffron_thyme_* test cases.
%
% The companion generator generate_saffron_thyme_refs.m covers the detection
% stage and the propagation with rectangular pulses. The six cases here are
% transcribed from EasySpin's own tests/saffron_thyme_*.m, four of which use
% quartersin/linear chirps, so they exercise what the rectangular suite
% cannot: Pulse.Type, Pulse.trise, swept Pulse.Frequency, a pulse-parameter
% sweep along an indirect dimension (Exp.Dim2 = {'p1.Flip' ...}), tensor frame
% tilts (Sys.gFrame / Sys.AFrame) against an equivalent sample tilt, and a
% predefined sequence detected over a transient window.
%
% Differences from EasySpin's test files, both deliberate:
%
%   * Opt.IntTimeStep is stated explicitly. EasySpin's default here is
%     3.2e-6 us (Nyquist/50, rounded), which is finer than needed for the
%     comparison and makes both the Python suite and the stored references
%     large. shaped_spinecho_fine repeats one case at 5e-6 us to show the
%     agreement does not depend on the step. Leaving the step implicit would
%     also let the two codes' defaults drift apart unnoticed, which was one of
%     the harness faults found in the 0.5.0 audit.
%   * Opt.SimulationMode is stated even where EasySpin's test relies on
%     Exp.DetWindow to force the thyme engine, for the same reason.
%
% Sys.g uses EasySpin's gfree in the framerotations case, as EasySpin's test
% does; the Python side uses torchspin.constants.GFREE, never a rounded
% 2.00232 -- a 3.5e-7 error in g was once mistaken for a simulator defect.
%
% Run from the repository root with EasySpin on the MATLAB path:
%
%   matlab -nodisplay -batch "run('tests/data/generate_saffron_thyme_shaped_refs.m')"

clear data

% Parallel pools are not needed here and fail on license-restricted hosts;
% saffron's orientation loop is a parfor, which falls back to serial.
try
  ps = parallel.Settings;
  ps.Pool.AutoCreate = false;
catch
end

% The chirps sweep +-300 MHz about a 34.78 GHz carrier, so their Nyquist time
% is 1.6e-4 us; 2e-5 us is 8x below it. The two cases without chirps run at
% 9.3 GHz with rectangular pulses, where the Nyquist time is 2.2e-4 us, so a
% coarser step is equally valid there and keeps the stored transients small.
INTTIMESTEP = 2e-5;        % us, chirp cases
INTTIMESTEP_RECT = 1e-4;   % us, rectangular cases

% =============================================================================
% Cases 1-4: Q-band chirp spin echo (saffron_thyme_spinecho, _relaxation,
% _simfreq, _indirectdimensions share this spin system and pulse pair)
% =============================================================================
SysQ = struct('S',0.5,'g',[2.00906 2.0062 2.0023],'A',[11.5 11.5 95], ...
              'Nucs','14N','lwpp',10);

Chirp90 = struct('Type','quartersin/linear','trise',0.030,'tp',0.200, ...
                 'Flip',pi/2,'Frequency',[-300 300]);
Chirp180 = struct('Type','quartersin/linear','trise',0.030,'tp',0.100, ...
                  'Flip',pi,'Frequency',[-300 300]);

ExpQ = struct('Field',1240,'mwFreq',34.78);
ExpQ.Sequence = {Chirp90 0.25 Chirp180 0.25};
ExpQ.DetWindow = [-0.1 0.1] + Chirp180.tp;
ExpQ.DetPhase = 0;

OptQ = struct('GridSize',7,'SimulationMode','thyme','IntTimeStep',INTTIMESTEP);

% --- 1. The chirp spin echo itself (saffron_thyme_spinecho).
[x,y] = saffron(SysQ,ExpQ,OptQ);
data.shaped_spinecho = struct('x',x,'y',y);

% --- 2. The same echo at a four times finer integration step, so that the
%        agreement cannot be an artefact of the step size.
OptFine = OptQ; OptFine.IntTimeStep = 5e-6;
[x,y] = saffron(SysQ,ExpQ,OptFine);
data.shaped_spinecho_fine = struct('x',x,'y',y);

% --- 3. Relaxation during the chirps (saffron_thyme_relaxation). This routes
%        the propagation through the Liouvillian, where the commutator
%        superoperator's vectorization convention matters.
SysRelax = SysQ; SysRelax.T1 = 2; SysRelax.T2 = 0.5;
[x,y] = saffron(SysRelax,ExpQ,OptQ);
data.shaped_relaxation = struct('x',x,'y',y);

% --- 4. Explicit simulation frame (saffron_thyme_simfreq). Opt.SimFreq = 15
%        puts the frame far from the automatic floor(mwFreq-2) = 32 GHz, so
%        the pulse carriers and the detection translation all shift.
ExpSim = ExpQ;
ExpSim.DetWindow = [-0.05 0.05] + Chirp180.tp;
OptSim = OptQ; OptSim.SimFreq = 15;
[x,y] = saffron(SysQ,ExpSim,OptSim);
data.shaped_simfreq = struct('x',x,'y',y);

% --- 5/6. Indirect dimensions (saffron_thyme_indirectdimensions): a delay
%        sweep, then a delay sweep crossed with a *pulse parameter* sweep
%        (Exp.Dim2 = {'p1.Flip' pi/3}), which re-generates the first chirp's
%        waveform at every point of that dimension.
ExpDim = ExpQ;
ExpDim.nPoints = 2;
ExpDim.Dim1 = {'d1,d2' 0.5};
[x,y] = saffron(SysQ,ExpDim,OptQ);
data.shaped_dim1 = struct('x',{x},'y',y);

ExpDim2 = ExpDim;
ExpDim2.nPoints = [3 2];
ExpDim2.Dim1 = {'d1,d2' 0.5};
ExpDim2.Dim2 = {'p1.Flip' pi/3};
[x,y] = saffron(SysQ,ExpDim2,OptQ);
data.shaped_dim2 = struct('x',{x},'y',y);

% =============================================================================
% Case 7/8: tensor frame tilts vs an equivalent sample tilt
% (saffron_thyme_framerotations). Rectangular pulse, but the only case that
% exercises Sys.gFrame / Sys.AFrame in the real-pulse route: a system whose
% tensors are tilted in the molecular frame, and an untilted system whose
% *sample* is tilted by the same angles, must give the same signal.
% =============================================================================
angles = [pi/3 pi/5 0];

SysF = struct('S',0.5,'J',0,'Nucs','1H','g',gfree + [0 0.005 0.01], ...
              'A',[5 2 1]);
SysTilted = SysF; SysTilted.gFrame = angles; SysTilted.AFrame = angles;

PulseF = struct('tp',0.01,'Flip',pi);
ExpF = struct('Field',332,'mwFreq',9.3);
ExpF.Sequence = {PulseF};
ExpF.DetWindow = [0 0.3];
ExpF.DetPhase = 0;
ExpF.MolFrame = [0 0 0];
OptF = struct('GridSize',7,'SimulationMode','thyme','IntTimeStep',INTTIMESTEP_RECT);

% Tensor tilts, no sample tilt
ExpF.SampleFrame = [0 0 0];
[x,y] = saffron(SysTilted,ExpF,OptF);
data.frames_tensor_tilt = struct('x',x,'y',y);

% No tensor tilts, sample tilted by the same angles
ExpF.SampleFrame = angles;
[x,y] = saffron(SysF,ExpF,OptF);
data.frames_sample_tilt = struct('x',x,'y',y);

% =============================================================================
% Case 9/10: predefined 3pESEEM detected over a *transient* window, single
% crystal (saffron_thyme_crystalorientations). The rectangular suite only
% detects predefined sequences at a single point, so this is the one case
% where Exp.DetWindow shortens the last delay of a template-built sequence.
% =============================================================================
SysC = struct('S',0.5,'g',1.898468237763169,'Nucs','1H','A_',[5 2]);

ExpC = struct('Sequence','3pESEEM','Field',350,'mwFreq',9.3);
ExpC.dt = 0.01;
ExpC.tau = 0.1;
ExpC.T = 0.5;
ExpC.nPoints = 10;
ExpC.DetWindow = [-0.05 0.05];
ExpC.DetPhase = 0;
ExpC.MolFrame = [0 0 0];
OptC = struct('SimulationMode','thyme','IntTimeStep',INTTIMESTEP_RECT);

ExpC.SampleFrame = [0 -pi/2 0];
[x,y] = saffron(SysC,ExpC,OptC);
data.crystal_3peseem_one = struct('x',{x},'y',y);

ExpC.SampleFrame = [0 -pi/2 0; 0 0 0];
[x,y] = saffron(SysC,ExpC,OptC);
data.crystal_3peseem_two = struct('x',{x},'y',y);

outFile = fullfile(fileparts(mfilename('fullpath')), ...
                   'saffron_thyme_shaped_refs.mat');
save(outFile,'data','-v7');
fprintf('wrote %s\n', outFile);
