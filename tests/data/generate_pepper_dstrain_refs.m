% D-strain spectrum references for PORT_SPEC_2 item 1.3 (pepper's DStrain
% accumulation for many-transition systems). Mirrors EasySpin tests
% pepper_dstrain_explicit, pepper_dstraincorrelated, pepper_multidstrain.
clear all
fprintf('Generating pepper D-strain references...\n');
cases = {};

% pepper_dstrain_explicit (S=1, axial D, D strain only, absorption)
Sys = struct('S',1,'g',2,'D',200,'DStrain',50,'lwpp',0.2);
Exp = struct('mwFreq',9.517,'CenterSweep',[340 30],'Harmonic',0);
Opt = struct('GridSize',121,'Verbosity',0);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','dstrain_explicit','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

% pepper_dstraincorrelated: DStrainCorr = 0, +1, -1
Sys = struct('S',1,'D',[800 80],'DStrain',[100 33],'lwpp',1);
Exp = struct('mwFreq',9.5,'CenterSweep',[340 100]);
Opt = struct('GridSize',15,'Verbosity',0);
for r = [0 1 -1]
  Sys.DStrainCorr = r;
  [B,spc] = pepper(Sys,Exp,Opt);
  cases{end+1} = struct('name',sprintf('dstraincorr_%+d',r),'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
end

% same system at finer grids (separates coarse-grid interpolation from the strain physics)
for gs = [31 61]
  for r = [0 1]
    Sys.DStrainCorr = r; Opt = struct('GridSize',gs,'Verbosity',0);
    [B,spc] = pepper(Sys,Exp,Opt);
    cases{end+1} = struct('name',sprintf('dstraincorr_%+d_gs%d',r,gs),'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
  end
end
% and without the strain at all (isolates the D-strain contribution)
Sys0 = rmfield(Sys,'DStrainCorr'); Sys0.DStrain = 0; Opt = struct('GridSize',15,'Verbosity',0);
[B,spc] = pepper(Sys0,Exp,Opt);
cases{end+1} = struct('name','dstraincorr_nostrain_gs15','Sys',Sys0,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

% pepper_multidstrain: two S=1 with different g and D, weak coupling
Sys = struct('S',[1 1],'g',[2 2.5],'D',[400 700],'ee',1,'lwpp',1);
Sys.DStrain = [30 10; 100 20];
Exp = struct('mwFreq',9.5,'Range',[200 400]);
Opt = struct('GridSize',[19 4],'Verbosity',0);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','multidstrain','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Opt = struct('GridSize',31,'Verbosity',0);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','multidstrain_31','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

% two coupled triplets with correlated D/E strain (from strain_ref_multielectron), finer grid
Sys = struct('S',[1 1],'g',[2.0 2.0]);
Sys.D = [500 60; 700 -90]; Sys.DStrain = [50 10; 40 20]; Sys.DStrainCorr = [0.5 -0.3]; Sys.ee = 50; Sys.lw = 0;
Exp = struct('mwFreq',9.5,'Range',[200 480],'nPoints',1024,'Harmonic',1);
Opt = struct('GridSize',61,'Verbosity',0);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','two_triplets_dstrain_corr_61','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_dstrain.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
