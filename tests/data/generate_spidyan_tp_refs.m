% spidyan: 1D indirect dimension varying the pulse length tp (PORT_SPEC_2 item 4.2)
clear all
Sys.S = 1/2; Sys.ZeemanFreq = 33.500;
Pulse.Type = 'rectangular'; Pulse.tp = 0.1; Pulse.Flip = pi;
Exp.Sequence = {Pulse 0.5 Pulse}; Exp.Field = 1240; Exp.mwFreq = 33.5;
Exp.DetSequence = 1; Exp.nPoints = 3; Exp.Dim1 = {'p1.tp' 0.02};
Exp.DetPhase = 0; Exp.DetOperator = {'z1'};
Opt.IntTimeStep = 0.0001; Opt.SimFreq = 32;
[t, signal] = spidyan(Sys,Exp,Opt);
if iscell(signal)
  n = numel(signal); L = max(cellfun(@numel,signal)); sig = nan(n,L); tax = nan(n,L);
  for k = 1:n, sig(k,1:numel(signal{k})) = signal{k}; tax(k,1:numel(t{k})) = t{k}; end
else
  sig = signal; tax = t;
end
data.t = tax; data.sig_re = real(sig); data.sig_im = imag(sig);
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'spidyan_onedimensional_tp.mat'),'data','-v7');
fprintf('saved spidyan_onedimensional_tp.mat (%dx%d)\n', size(sig,1), size(sig,2));
