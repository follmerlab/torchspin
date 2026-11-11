% Powder spectra that go through the closed-phi grid interpolation (B25, 0.8.9):
% EasySpin gridinterp.m:184 resamples each theta row of the triangular grid
% with esspline1d(...,1,nc) (zero end slopes). Rhombic g at the default
% GridSize [19 4] and at [31 4], field sweeps at Harmonic 0 and 1 with
% Gaussian and Lorentzian widths, an HStrain field sweep, a 14N hyperfine
% field sweep, a frequency sweep, and fixed-field salt (Method 'perturb1')
% at [19 4] and [31 4]; also a central difference d spc/d g_z (rhombic, [19 4]). Reference for
% torchspin/tests/test_closedphi_interp_matlab_validation.py.
% Run from tests/data with EasySpin on the path:  >> generate_closedphi_interp_refs
clear all; cases = {};
SysG = struct('S',1/2,'g',[2 2.05 2.1],'lw',[1 0]);
SysL = struct('S',1/2,'g',[2 2.05 2.1],'lw',[0 0.8]);
Exp0 = struct('mwFreq',9.5,'Range',[320 345],'nPoints',1024,'Harmonic',0);
Exp1 = Exp0; Exp1.Harmonic = 1;
G19 = struct('GridSize',[19 4]); G31 = struct('GridSize',[31 4]);
cases{end+1} = mk('gauss_h0_19',SysG,Exp0,G19);
cases{end+1} = mk('gauss_h1_19',SysG,Exp1,G19);
cases{end+1} = mk('lorentz_h0_19',SysL,Exp0,G19);
cases{end+1} = mk('lorentz_h1_19',SysL,Exp1,G19);
cases{end+1} = mk('gauss_h1_31',SysG,Exp1,G31);
cases{end+1} = mk('lorentz_h1_31',SysL,Exp1,G31);
% default options (GridSize [19 4] is EasySpin's default)
cases{end+1} = mk('gauss_h1_default',SysG,Exp1,struct());
% HStrain: positions G3, widths L1
cases{end+1} = mk('hstrain_h1_19',struct('S',1/2,'g',[2 2.05 2.1],'HStrain',[60 80 100]),Exp1,G19);
% frequency sweep, Lorentzian 30 MHz
ExpF = struct('Field',340,'mwRange',[9.4 10.1],'nPoints',1024,'Harmonic',1);
cases{end+1} = mk('freq_lorentz_h1_19',struct('S',1/2,'g',[2 2.05 2.1],'lw',[0 30]),ExpF,G19);
% several transition slots: rhombic g with a coaxial rhombic 14N hyperfine
ExpN = struct('mwFreq',9.5,'Range',[315 350],'nPoints',2048,'Harmonic',1);
cases{end+1} = mk('n14_h1_19',struct('S',1/2,'g',[2 2.05 2.1],'Nucs','14N','A',[20 30 50],'lw',[0.5 0]),ExpN,G19);
% fixed-field salt (ENDOR), rhombic g and A, interpolated D2h grid, first-order
% perturbation (torchspin salt's path for one electron without Q)
salts = {};
SysE = struct('S',1/2,'g',[2 2.05 2.1],'Nucs','1H','A',[3 6 12],'lwEndor',0.1);
ExpE = struct('Field',331,'mwFreq',9.5,'Range',[5 25],'nPoints',1024);
P19 = struct('GridSize',[19 4],'Method','perturb1');
salts{end+1} = mks('salt_1H_19',SysE,ExpE,P19);
salts{end+1} = mks('salt_1H_31',SysE,ExpE,struct('GridSize',[31 4],'Method','perturb1'));
% central difference d spc / d g_z of the gauss_h1_19 spectrum (h = 1e-4)
h = 1e-4; SysP = SysG; SysP.g(3) = SysP.g(3) + h; SysM = SysG; SysM.g(3) = SysM.g(3) - h;
[~,yp] = pepper(SysP,Exp1,struct('GridSize',[19 4],'Verbosity',0));
[~,ym] = pepper(SysM,Exp1,struct('GridSize',[19 4],'Verbosity',0));
dgz = struct('h',h,'dspc',(yp-ym)/(2*h));
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_closedphi_interp.mat'),'cases','salts','dgz','-v7');
fprintf('Done: %d pepper cases, %d salt cases\n',numel(cases),numel(salts));

function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end

function c = mks(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = salt(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
