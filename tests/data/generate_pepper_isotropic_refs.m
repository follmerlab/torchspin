% Isotropic systems (O3 grid, one orientation) in pepper (B66): every line is
% drawn with an exact template at its position (EasySpin ~anisotropicSpectrum).
% Reference for torchspin/tests/test_pepper_autograd_o3.py, which runs the cases
% through pepper and through pepper_autograd.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
Exp = struct('mwFreq',9.5,'Range',[320 360],'nPoints',1024);
E = Exp; E.Harmonic = 1;
cases{end+1} = mk('S12_gauss_h1',struct('S',1/2,'g',2,'lw',[1 0]),E,struct());
E = Exp; E.Harmonic = 0;
cases{end+1} = mk('H1_lorentz_h0',struct('S',1/2,'g',2,'Nucs','1H','A',30,'lw',[0 0.5]),E,struct());
E = Exp; E.Harmonic = 1;
cases{end+1} = mk('N14_voigt_h1',struct('S',1/2,'g',2.005,'Nucs','14N','A',45,'lw',[0.3 0.3]),E,struct());
E = Exp; E.Harmonic = 2; E.Range = [250 400];
cases{end+1} = mk('Cu63_gauss_h2',struct('S',1/2,'g',2.1,'Nucs','63Cu','A',300,'lw',[1.5 0]),E,struct());
E = Exp; E.Harmonic = 1;
cases{end+1} = mk('H1_gauss_h1_O3explicit',struct('S',1/2,'g',2,'Nucs','1H','A',30,'lw',[0.5 0]),E,struct('GridSymmetry','O3'));
E = Exp; E.Harmonic = 1; E.Temperature = 5;
cases{end+1} = mk('S1_gauss_h1_T5',struct('S',1,'g',2,'lw',[1 0]),E,struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_isotropic.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
