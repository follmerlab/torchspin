% pepper with non-equilibrium populations (Sys.initState) and strain broadening
% (HStrain, DStrain, gStrain). Strains route pepper through its per-line
% summation, which must keep emissive (negative) intensities.
% Saved to ref_pepper_initstate_strain.mat (-v7) as a cell array `cases`, in the
% layout of ref_pepper_features2.mat (initPops / initBasis next to Sys).
clear all; cases = {};
function c = mkinit(name,Sys,Exp,Opt,pops,basis)
  Opt.Verbosity = 0; Sys.initState = {pops,basis}; [B,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
  c.Sys = rmfield(c.Sys,'initState'); c.initPops = pops; c.initBasis = basis;
end
Exp = struct('mwFreq',9.7,'Range',[290 400],'Harmonic',0,'nPoints',1024);
Tr = struct('S',1,'g',2.003,'D',[1500 300],'lwpp',[1 0],'HStrain',[30 30 30]);
cases{end+1} = mkinit('trip_HStrain_xyz',Tr,Exp,struct(),[0.1 0.3 0.6],'xyz');
Exp1 = Exp; Exp1.Harmonic = 1;
cases{end+1} = mkinit('trip_HStrain_xyz_h1',Tr,Exp1,struct(),[0.1 0.3 0.6],'xyz');
Tr = struct('S',1,'g',2.003,'D',[-1000 150],'lwpp',[0.5 0],'DStrain',[60 20]);
cases{end+1} = mkinit('trip_DStrain_zerofield',Tr,Exp,struct(),[0.2 0.2 0.6],'zerofield');
Tr = struct('S',1,'g',[2.002 2.003 2.004],'D',[900 280],'DFrame',[0.3 1.1 0.7],'lwpp',[1 0],'gStrain',[0.002 0.002 0.004]);
cases{end+1} = mkinit('trip_gStrain_xyz_DFrame',Tr,Exp,struct(),[0.2 0.5 0.3],'xyz');
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_initstate_strain.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
