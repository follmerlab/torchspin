% Generate MATLAB references for saffron's real-pulse ("thyme") route.
%
% Before this suite the real-pulse route had no MATLAB cross-validation at
% all -- only internal-consistency tests, which cannot catch an error shared
% by both sides. Every case below therefore exercises the detection stage
% EasySpin runs after propagation (saffron.m 1884-1940) and that torchspin
% used to skip entirely: the Exp.OriWeights orientation sum, Exp.DetPhase,
% Exp.DetIntegrate echo integration, down-conversion through
% signalprocessing.m, and the Exp.DetWindow time axis.
%
% All pulses are rectangular, so no pulse shaping is needed. Each case states
% every option that affects the result -- Opt.SimulationMode, Opt.GridSize,
% Opt.IntTimeStep, Opt.SimFreq, Exp.DetWindow, Exp.DetPhase,
% Exp.DetIntegrate -- because three of the harness faults found in the 0.5.0
% pulse-EPR audit were unstated options rather than wrong code. The matching
% Python tests assert the relative L2 error, never a cosine.
%
% Opt.IntTimeStep is deliberately pinned rather than left to EasySpin's
% default: the default is derived from the Nyquist criterion and rounded to
% two significant digits, so it would silently differ between the two codes
% if either changed that formula. 1e-4 µs is coarse (EasySpin warns about it)
% but both codes integrate identically with it, which is what is being
% tested; eseem2p_fine repeats a case at 2e-5 µs to show agreement does not
% depend on the step.
%
% Sys.g uses EasySpin's gfree where a free-electron g is wanted; the Python
% side uses torchspin.constants.GFREE, not a rounded 2.00232.
%
% Run from the repository root with EasySpin on the MATLAB path:
%
%   matlab -nodisplay -batch "run('tests/data/generate_saffron_thyme_refs.m')"

clear data

% Parallel pools are not needed here and fail on license-restricted hosts;
% saffron's orientation loop is a parfor, which falls back to serial.
try
  ps = parallel.Settings;
  ps.Pool.AutoCreate = false;
catch
end

% Spin systems ---------------------------------------------------------------
SysG = struct('S',0.5,'g',[2.00906 2.0062 2.0023]);
Sys1H = struct('S',0.5,'g',[2.00906 2.0062 2.0023],'Nucs','1H', ...
               'A',[3.0 3.0 8.0]);
Sys14N = struct('S',0.5,'g',[2.00906 2.0062 2.0023],'Nucs','14N', ...
                'A',[11.5 11.5 95],'Q',[0.8 0.5]);
SysRelax = Sys1H; SysRelax.T1 = 2; SysRelax.T2 = 0.5;

% Rectangular pulses ---------------------------------------------------------
P90 = struct('tp',0.010,'Flip',pi/2);
P180 = struct('tp',0.020,'Flip',pi);

% Common options -------------------------------------------------------------
Opt = struct('SimulationMode','thyme','IntTimeStep',1e-4,'GridSize',7);

% --- 1. Free induction decay, powder: the simplest transient. Checks the
%        simulation-frame g shift, the pulse carrier at Exp.mwFreq, the
%        orientation sum with the 4*pi grid weights (no renormalization) and
%        the down-conversion back to the detection frequency.
Exp = struct('Field',332,'mwFreq',9.3);
Exp.Sequence = {P90 0.05};
Exp.DetWindow = [0 0.05];
Exp.DetPhase = 0;
[x,y] = saffron(SysG,Exp,Opt);
data.fid_powder = struct('x',x,'y',y);

% --- 2. The same transient for a single crystal (P1, one site), which
%        removes the powder grid from the comparison.
ExpC = Exp;
ExpC.CrystalSymmetry = 'P1';
ExpC.MolFrame = [0 0 0];
ExpC.SampleFrame = [0 pi/5 0];
[x,y] = saffron(SysG,ExpC,Opt);
data.fid_crystal = struct('x',x,'y',y);

% --- 3. Two-pulse echo with a 1H nucleus, detected over a window centered on
%        the echo. The window is applied by shifting the last delay, not by
%        slicing afterwards, so getting it wrong moves the echo.
ExpEcho = struct('Field',332,'mwFreq',9.3);
tau = 0.15;
ExpEcho.Sequence = {P90 tau P180 tau};
% For rectangular pulses the echo refocuses at the end of the last delay,
% so the window is centered on zero (EasySpin's own thyme tests offset it by
% the pulse length because their chirps delay the echo). A negative start
% also exercises the branch that shortens the preceding free evolution.
ExpEcho.DetWindow = [-0.03 0.03];
ExpEcho.DetPhase = 0;
[x,y] = saffron(Sys1H,ExpEcho,Opt);
data.echo_window = struct('x',x,'y',y);

% --- 4. The same echo with a nonzero detection phase.
ExpPhase = ExpEcho;
ExpPhase.DetPhase = pi/4;
[x,y] = saffron(Sys1H,ExpPhase,Opt);
data.echo_detphase = struct('x',x,'y',y);

% --- 5. The same echo integrated (Exp.DetIntegrate), which collapses the
%        transient to one point and switches detection to single-point.
ExpInt = ExpEcho;
ExpInt.DetIntegrate = true;
[x,y] = saffron(Sys1H,ExpInt,Opt);
data.echo_integrate = struct('x',x,'y',y);

% --- 6. Single-point detection with an indirect dimension: a real-pulse
%        two-pulse ESEEM trace, one detected point per increment. Exp.tau is
%        nonzero so the modulation, not just the DC level, is compared.
ExpSP = struct('Field',332,'mwFreq',9.3);
ExpSP.Sequence = {P90 tau P180 tau};
ExpSP.DetWindow = 0;             % scalar -> single-point detection
ExpSP.DetPhase = 0;
ExpSP.nPoints = 16;
ExpSP.Dim1 = {'d1,d2' 0.02};
[x,y] = saffron(Sys1H,ExpSP,Opt);
data.eseem2p_singlepoint = struct('x',x,'y',y);

% --- 7. The same case at a five times finer integration step, to show the
%        agreement is not an artefact of a coarse time step.
OptFine = Opt; OptFine.IntTimeStep = 2e-5;
[x,y] = saffron(Sys1H,ExpSP,OptFine);
data.eseem2p_fine = struct('x',x,'y',y);

% --- 8. A quadrupolar 14N with a large anisotropic hyperfine coupling, so
%        the nuclear sublevel structure (not just the electron) is tested.
[x,y] = saffron(Sys14N,ExpSP,Opt);
data.eseem2p_14n = struct('x',x,'y',y);

% --- 9. Explicit simulation frame: Opt.SimFreq fixes the frame shift instead
%        of letting s_sequencer pick floor(mwFreq-2).
OptSim = Opt; OptSim.SimFreq = 7.5;
[x,y] = saffron(Sys1H,ExpEcho,OptSim);
data.echo_simfreq = struct('x',x,'y',y);

% --- 10. Exp.DetSequence instead of Exp.DetWindow, with a zero-length final
%         delay. s_sequencer resolves that to single-point detection
%         (sum(DetSequence) == 1 and the detected event is a zero-length
%         delay), which is the other way into the single-point contract.
%         A *transient* Exp.DetSequence cannot be referenced here: saffron.m
%         line 1937 reads Exp.DetWindow(1) unconditionally once detection is
%         not single-point, so EasySpin itself errors with
%         'Unrecognized field name "DetWindow"'.
ExpDS = struct('Field',332,'mwFreq',9.3);
ExpDS.Sequence = {P90 tau P180 tau 0};
ExpDS.DetSequence = [0 0 0 0 1];
ExpDS.DetPhase = 0;
ExpDS.nPoints = 8;
ExpDS.Dim1 = {'d1,d2' 0.02};
[x,y] = saffron(Sys1H,ExpDS,Opt);
data.eseem2p_detsequence = struct('x',x,'y',y);

% --- 11. Predefined 3pESEEM through the real-pulse engine, which EasySpin
%         allows (s_predefinedexperiments builds the sequence) while the fast
%         algorithm rejects it. Exercises the built-in two-step phase cycles
%         on pulses 1 and 3.
Exp3p = struct('Field',332,'mwFreq',9.3,'Sequence','3pESEEM');
Exp3p.dt = 0.02;
Exp3p.tau = 0.1;
Exp3p.T = 0.08;
Exp3p.nPoints = 12;
Exp3p.DetWindow = 0;
Exp3p.DetPhase = 0;
[x,y] = saffron(Sys1H,Exp3p,Opt);
data.eseem3p_predefined = struct('x',x,'y',y);

% --- 12. Predefined 2pESEEM through the real-pulse engine, single crystal.
Exp2p = struct('Field',332,'mwFreq',9.3,'Sequence','2pESEEM');
Exp2p.dt = 0.02;
Exp2p.tau = 0.12;
Exp2p.nPoints = 12;
Exp2p.DetWindow = 0;
Exp2p.DetPhase = 0;
Exp2p.CrystalSymmetry = 'P1';
Exp2p.MolFrame = [0 0 0];
Exp2p.SampleFrame = [0 -pi/2 0];
[x,y] = saffron(Sys1H,Exp2p,Opt);
data.eseem2p_crystal = struct('x',x,'y',y);

% --- 13. Relaxation during the sequence (Sys.T1 / Sys.T2), which routes the
%         propagation through the Liouvillian rather than the propagator.
[x,y] = saffron(SysRelax,ExpSP,Opt);
data.eseem2p_relaxation = struct('x',x,'y',y);

% --- 14. Two indirect dimensions: predefined HYSCORE through the real-pulse
%         engine, with unequal nPoints so a transposed 2D buffer cannot hide.
ExpHys = struct('Field',332,'mwFreq',9.3,'Sequence','HYSCORE');
ExpHys.dt = 0.03;
ExpHys.tau = 0.1;
ExpHys.t1 = 0.02;
ExpHys.t2 = 0.03;
ExpHys.nPoints = [4 6];
ExpHys.DetWindow = 0;
ExpHys.DetPhase = 0;
[x,y] = saffron(Sys1H,ExpHys,Opt);
data.hyscore_predefined = struct('x',{x},'y',y);

% --- 15. A transient over an indirect dimension: the output keeps both the
%         increment axis and the transient axis.
ExpTrans = ExpSP;
ExpTrans.DetWindow = [-0.02 0.02];
ExpTrans.nPoints = 4;
ExpTrans.Dim1 = {'d1,d2' 0.02};
[x,y] = saffron(Sys1H,ExpTrans,Opt);
data.echo_transient_dim = struct('x',{x},'y',y);

% --- 16. Two stacked crystal orientations. EasySpin gives every crystal
%         orientation the weight 4*pi and *sums* without renormalizing, so
%         this must equal the sum -- not the mean -- of the two single-site
%         runs. Pinning it in MATLAB settles the crystal normalization that
%         an internal consistency relation alone cannot.
Exp2pTwo = Exp2p;
Exp2pTwo.SampleFrame = [0 -pi/2 0; 0 0 0];
[x,y] = saffron(Sys1H,Exp2pTwo,Opt);
data.eseem2p_crystal_two = struct('x',x,'y',y);

Exp2pB = Exp2p;
Exp2pB.SampleFrame = [0 0 0];
[x,y] = saffron(Sys1H,Exp2pB,Opt);
data.eseem2p_crystal_b = struct('x',x,'y',y);

outFile = fullfile(fileparts(mfilename('fullpath')),'saffron_thyme_refs.mat');
save(outFile,'data','-v7');
fprintf('wrote %s\n', outFile);
