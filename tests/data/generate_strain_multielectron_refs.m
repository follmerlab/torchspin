% Multi-electron gStrain / DStrain reference spectra (pepper) for torchspin's
% strainwidth port of EasySpin resfields.m (non-simple g strain, per-electron
% getdstrainops with DFrame and DStrainCorr).
% Run from tests/data with EasySpin on the path.

clear all
fprintf('Generating multi-electron strain references...\n');
cases = {};

% Case 1: two S=1/2, rhombic g on both, g strain on both, one tilted g frame
Sys = struct('S',[1/2 1/2]);
Sys.g = [2.00 2.05 2.15; 1.98 2.02 2.10];
Sys.gFrame = [0 0 0; pi/6 pi/4 pi/3];
Sys.gStrain = [0.01 0.02 0.03; 0.02 0.01 0.015];
Sys.ee = 300;   % MHz, isotropic
Sys.lw = 0;
Exp = struct('mwFreq',9.5,'Range',[290 380],'nPoints',1024,'Harmonic',1);
Opt = struct('GridSize',31,'Verbosity',0);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','two_spinhalf_gstrain_both','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

% Case 2: two S=1/2, g strain only on electron 2 (weighting must pick electron 2)
Sys = struct('S',[1/2 1/2]);
Sys.g = [2.00 2.00 2.00; 2.00 2.10 2.20];
Sys.gStrain = [0 0 0; 0.01 0.02 0.03];
Sys.ee = 1000;  % MHz
Sys.lw = 0;
Exp = struct('mwFreq',9.5,'Range',[280 400],'nPoints',1024,'Harmonic',1);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','two_spinhalf_gstrain_second','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

% Case 3: S=1, rhombic D in a tilted frame, D and E strain
Sys = struct('S',1,'g',2.0);
Sys.D = [600 80];
Sys.DFrame = [pi/5 pi/3 pi/7];
Sys.DStrain = [60 15];
Sys.lw = 0;
Exp = struct('mwFreq',9.5,'Range',[250 430],'nPoints',1024,'Harmonic',1);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','triplet_dstrain_tilted','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

% Case 4: two S=1, per-electron D strain with D/E correlation
Sys = struct('S',[1 1],'g',[2.0 2.0]);
Sys.D = [500 60; 700 -90];
Sys.DStrain = [50 10; 40 20];
Sys.DStrainCorr = [0.5 -0.3];
Sys.ee = 50;
Sys.lw = 0;
Exp = struct('mwFreq',9.5,'Range',[200 480],'nPoints',1024,'Harmonic',1);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','two_triplets_dstrain_corr','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

% Case 5: S=1 with correlated D/E strain, no tilt (isolates DStrainCorr)
Sys = struct('S',1,'g',2.0);
Sys.D = [800 150];
Sys.DStrain = [80 30];
Sys.DStrainCorr = 0.8;
Sys.lw = 0;
Exp = struct('mwFreq',9.5,'Range',[250 430],'nPoints',1024,'Harmonic',1);
[B,spc] = pepper(Sys,Exp,Opt);
cases{end+1} = struct('name','triplet_dstrain_corr','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'strain_ref_multielectron.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
