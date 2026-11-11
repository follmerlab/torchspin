% Final Sys.lwEndor convolution in salt (torchspin bug B24, fixed in 0.8.11):
% a Gaussian or Lorentzian FWHM of at most two increments with harmonic 0 is
% not convolved (salt.m:884-911). Reference for
% torchspin/tests/test_salt_narrowlw_matlab_validation.py.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
% Saves salt_narrowlw_refs.mat with cell array `cases`, one struct per case:
%   .name, .Sys, .Exp, .Opt  inputs (fixed-field salt, Opt.Method = 'perturb1')
%   .x, .spc                 rf axis (MHz) and spectrum
% Gaussian widths 0, 1.5, 2.0, 2.5 and 10 increments (increment 0.02 MHz).
% The spectrum is zero at both ends of the rf range, so salt.m's range
% padding (salt.m:858-882) is not used.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = salt(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x(:)','spc',spc(:)');
end
Sys = struct('S',1/2,'g',[2.0 2.05 2.1],'Nucs','1H','A',[5 8 12]);
Exp = struct('Field',340,'Range',[2 28],'nPoints',1301);   % increment 0.02 MHz
Opt = struct('GridSize',31,'Method','perturb1');
names = {'lw_0','lw_1p5dx','lw_2p0dx','lw_2p5dx','lw_10dx'};
widths = [0 0.03 0.04 0.05 0.2];
for k = 1:numel(widths)
  Sys.lwEndor = [widths(k) 0];
  cases{end+1} = mk(names{k},Sys,Exp,Opt);
end
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'salt_narrowlw_refs.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
