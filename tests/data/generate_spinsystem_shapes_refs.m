% SpinSystem input shorthands that validatespinsys.m expands (B62, B63): each
% case is given to pepper in the short form; torchspin must expand it the same
% way. Reference for torchspin/tests/test_spinsystem_shapes.py.
%   g_axial           Sys.g = [gperp gpar] (1x2 = [nElectrons 2])       -> [gperp gperp gpar]  (141-143)
%   gStrain_xy        Sys.gStrain = [x y] (1x2)                          -> [x x y]             (822-823)
%   HStrain_xy        Sys.HStrain = [x y]                                -> [x x y]             (888)
%   A_axial_14N       Sys.A = [Aperp Apar] (1x2 = [nNuclei 2*nElectrons]) -> [Aperp Aperp Apar] (605-611)
%   A_iso_per_e_2e    Sys.A = [A1 A2], 1 nucleus, 2 electrons ([nNuclei nElectrons]) (602-604)
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
Exp = struct('mwFreq',9.5,'Range',[300 350],'nPoints',1024);
cases{end+1} = mk('g_axial',struct('S',1/2,'g',[2.0 2.2],'lw',[1 0]),Exp,struct());
cases{end+1} = mk('gStrain_xy',struct('S',1/2,'g',[2 2.05 2.1],'gStrain',[0.004 0.008]),Exp,struct());
cases{end+1} = mk('HStrain_xy',struct('S',1/2,'g',[2 2.05 2.1],'HStrain',[30 60]),Exp,struct());
cases{end+1} = mk('A_axial_14N',struct('S',1/2,'g',[2.0 2.0 2.1],'Nucs','14N','A',[20 90],'lw',[0.5 0]),Exp,struct());
Sys2 = struct('S',[1/2 1/2],'g',[2.0 2.0 2.0; 2.1 2.1 2.1],'ee',200,'Nucs','1H','A',[30 60],'lw',[0.5 0]);
cases{end+1} = mk('A_iso_per_e_2e',Sys2,Exp,struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_spinsystem_shapes.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
