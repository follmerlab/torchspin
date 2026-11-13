% Range padding before the final line-width convolution (torchspin bug B34,
% fixed in 0.9.6): when the spectrum is nonzero at the first or last point of
% the sweep range, salt.m (858-882) and pepper.m (1171-1193) extend both ends
% by nPad = round(max([fwhmG fwhmL])/deltaX*Opt.PaddingMultiplier) copies of
% the edge columns before convolving, and crop afterwards. Reference for
% torchspin/tests/test_range_padding_matlab_validation.py.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
% Saves range_padding_refs.mat with cell array `cases`, one struct per case:
%   .name, .Sys, .Exp, .Opt   inputs (Opt without PaddingMultiplier: default 3)
%   .x                        sweep axis (MHz for salt; mT or GHz for pepper)
%   .spc                      spectrum, default Opt.PaddingMultiplier (3)
%   .spc_pm0                  Opt.PaddingMultiplier = 0 (no padding)
%   .spc_pm1                  Opt.PaddingMultiplier = 1
% Cases: fixed-field salt 1H (perturb1) on 10-19 MHz, lwEndor 0.2 MHz; salt
% 1H + 14N with Q (matrix) on 1.5-19 MHz, lwEndor 0.3 MHz; pepper frequency
% sweep, Lorentzian lw 40 MHz, Harmonic 1, 9.3-9.55 GHz; pepper field sweep
% with HStrain and a Gaussian lw on 333-342 mT; pepper Voigt field sweep on
% 325-337 mT (spectrum zero at the range ends before the convolution, so not
% padded: spc equals spc_pm0).
clear all; cases = {};
function c = mk(fun,name,Sys,Exp,Opt)
  Opt.Verbosity = 0;
  [x,spc] = fun(Sys,Exp,Opt);
  O = Opt; O.PaddingMultiplier = 0; [~,spc0] = fun(Sys,Exp,O);
  O = Opt; O.PaddingMultiplier = 1; [~,spc1] = fun(Sys,Exp,O);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x(:)','spc',spc(:)', ...
             'spc_pm0',spc0(:)','spc_pm1',spc1(:)');
end
% salt, first-order path
Sys = struct('S',1/2,'g',[2 2.05 2.1],'Nucs','1H','A',[5 8 12],'lwEndor',0.2);
Exp = struct('Field',330,'Range',[10 19],'nPoints',451);
Opt = struct('GridSize',[19 4],'Method','perturb1');
cases{end+1} = mk(@salt,'salt_1H_perturb1',Sys,Exp,Opt);
% salt, matrix path (quadrupole)
Sys = struct('S',1/2,'g',[2 2.05 2.1],'Nucs','1H,14N','A',[5 8 12; 2 3 4], ...
             'Q',[0 0 0; -0.3 -0.1 0.4],'lwEndor',0.3);
Exp = struct('Field',330,'Range',[1.5 19],'nPoints',501);
Opt = struct('GridSize',[19 4],'Method','matrix');
cases{end+1} = mk(@salt,'salt_1H14N_matrix',Sys,Exp,Opt);
% pepper frequency sweep, Lorentzian convolution with the harmonic
Sys = struct('S',1/2,'g',[2 2.05 2.1],'lw',[0 40]);
Exp = struct('Field',330,'mwRange',[9.3 9.55],'nPoints',601);
Opt = struct('GridSize',[31 4]);
cases{end+1} = mk(@pepper,'pepper_freq_L_h1',Sys,Exp,Opt);
% pepper field sweep, HStrain spectrum convolved with a Gaussian lw
% (all resonance fields inside the range; the strain lines reach the ends)
Sys = struct('S',1/2,'g',[2 2.01 2.02],'HStrain',[150 150 150],'lw',1.0);
Exp = struct('mwFreq',9.5,'Range',[333 342],'nPoints',451);
cases{end+1} = mk(@pepper,'pepper_strain_G_h1',Sys,Exp,Opt);
% pepper Voigt field sweep (not padded)
Sys = struct('S',1/2,'g',[2 2.05 2.1],'lw',[0.5 1.5]);
Exp = struct('mwFreq',9.5,'Range',[325 337],'nPoints',601);
cases{end+1} = mk(@pepper,'pepper_V_h1',Sys,Exp,Opt);
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'range_padding_refs.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
