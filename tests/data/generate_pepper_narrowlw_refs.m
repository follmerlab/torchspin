% Final Sys.lw convolution in pepper (B13): a line width of at most two
% increments with harmonic 0 is not convolved (pepper.m:1195-1223), and with
% a Voigt width the Lorentzian carries the harmonic and Exp.mwPhase
% (pepper.m:1152-1163). Reference for
% torchspin/tests/test_pepper_narrowlw_matlab_validation.py.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
% under-resolved frequency sweep: FWHM 11.8 MHz, increment 9.8 MHz (default Harmonic 0)
SysF = struct('S',1,'g',2,'D',[3000 300],'lwpp',10);
ExpF = struct('Field',340,'mwRange',[5 15],'nPoints',1024);
cases{end+1} = mk('triplet_freq_narrow',SysF,ExpF,struct());
cases{end+1} = mk('triplet_freq_narrow_12',SysF,ExpF,struct('Transitions',[1 2]));
% under-resolved field sweep, Harmonic 0: FWHM 0.03 mT, increment 0.024 mT
ExpB = struct('mwFreq',9.5,'Range',[320 345],'nPoints',1024,'Harmonic',0);
cases{end+1} = mk('field_narrow_gauss',struct('S',1/2,'g',[2 2.05 2.1],'lw',[0.03 0]),ExpB,struct());
cases{end+1} = mk('field_narrow_lorentz',struct('S',1/2,'g',[2 2.05 2.1],'lw',[0 0.03]),ExpB,struct());
% Voigt widths: harmonic and mwPhase go to the Lorentzian
ExpV = struct('mwFreq',9.5,'Range',[320 345],'nPoints',1024,'Harmonic',1);
cases{end+1} = mk('field_voigt_h1',struct('S',1/2,'g',[2 2.05 2.1],'lw',[1 0.8]),ExpV,struct());
ExpVP = ExpV; ExpVP.mwPhase = 0.3;
cases{end+1} = mk('field_voigt_h1_phase',struct('S',1/2,'g',[2 2.05 2.1],'lw',[1 0.8]),ExpVP,struct());
% narrow Gaussian + resolved Lorentzian, Harmonic 1: the Gaussian has harmonic 0 and is skipped
cases{end+1} = mk('field_voigt_narrowG_h1',struct('S',1/2,'g',[2 2.05 2.1],'lw',[0.03 0.8]),ExpV,struct());
ExpFV = struct('Field',340,'mwRange',[5 15],'nPoints',1024,'Harmonic',1);
cases{end+1} = mk('triplet_freq_voigt_h1',struct('S',1,'g',2,'D',[3000 300],'lw',[30 20]),ExpFV,struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_narrowlw.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
