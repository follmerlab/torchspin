% Position interpolation when some transition lacks a resonance at some knots
% (B31): EasySpin then interpolates the positions of EVERY transition
% linearly, L1 on 2-D grids (pepper.m:701, 763-772), not only those that
% carry NaNs. Reference for
% torchspin/tests/test_pepper_nanlinear_matlab_validation.py.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
% S=5/2 with D and a fourth-order B4 term; the range cuts off outer transitions
SysA = struct('S',5/2,'g',2,'D',300,'lw',[1 0]); SysA.B4 = [30 0 0 0 10 0 0 0 0];
ExpA = struct('mwFreq',9.5,'Range',[200 480],'nPoints',1024,'Harmonic',1);
cases{end+1} = mk('s52_b4_D2h_h1',SysA,ExpA,struct('GridSize',[19 4],'GridSymmetry','D2h'));
ExpA0 = ExpA; ExpA0.Harmonic = 0;
cases{end+1} = mk('s52_b4_D2h_h0',SysA,ExpA0,struct('GridSize',[19 4],'GridSymmetry','D2h'));
% open-phi grid (C2h): the same switch, but a residual remains (see the test)
cases{end+1} = mk('s52_b4_C2h_h1',SysA,ExpA,struct('GridSize',[19 4],'GridSymmetry','C2h'));
% rhombic S=5/2 (D2h from hamsymm), default grid, truncated range
SysB = struct('S',5/2,'g',2,'D',[400 100],'lw',[1.5 0]);
cases{end+1} = mk('s52_rhombic_h1',SysB,struct('mwFreq',9.5,'Range',[150 450],'nPoints',2048,'Harmonic',1),struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_nanlinear.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
