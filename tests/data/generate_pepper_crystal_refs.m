% Single-crystal pepper references (PORT_SPEC_2 item 3.1), mirroring EasySpin's
% pepper_crystal_* tests with fixed angles instead of rand.
clear all
cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [B,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
end
% crystal_molframe: gFrame vs MolFrame equivalence (space group 130)
g = [2.0 2.1 2.2]; gFrame = [30 40 78]*pi/180; Exp = struct('mwFreq',9.5,'Range',[290 350],'SampleFrame',[0.7 1.9 2.4],'CrystalSymmetry',130);
Sys = struct('g',g,'lw',0.5,'gFrame',gFrame); Exp.MolFrame = [0 0 0]; cases{end+1} = mk('molframe_gFrame',Sys,Exp,struct());
Sys.gFrame = [0 0 0]; Exp.MolFrame = gFrame; cases{end+1} = mk('molframe_MolFrame',Sys,Exp,struct());
% crystal_multiori (D2h, 4 orientations)
Sys = struct('g',[2 2.1 2.2],'gFrame',-[78 40 30]*pi/180,'lw',0.5); Exp = struct('mwFreq',9.5,'Range',[290 350],'CrystalSymmetry','D2h');
[phi,theta] = vec2ang([1 2 3; 1 -2 4; 0 0 1; 5 2 -3]); chi = zeros(size(phi(:))); Exp.SampleFrame = [-chi -theta(:) -phi(:)];
cases{end+1} = mk('multiori_D2h',Sys,Exp,struct());
% crystal_samplerot (rotation about lab x, 7 angles; sum and separate)
Sys = struct('g',[2 2.1 2.2],'lwpp',0.1); Exp = struct('MolFrame',[8 20 76]*pi/180,'CrystalSymmetry',1,'SampleFrame',[10 33 -8]*pi/180,'mwFreq',9.5,'Harmonic',0,'nPoints',1e4);
rho = linspace(0,pi,7); Exp.SampleRotation = {'x',rho}; Exp.Range = [290 350];
cases{end+1} = mk('samplerot_sum',Sys,Exp,struct());
Opt = struct('separate','orientations'); [B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','samplerot_separate','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
% crystal_sites (space group 34, all sites vs site 1)
Sys = struct('g',[2 2.1 2.2],'lwpp',1); Exp = struct('mwFreq',9.5,'Range',[200 400],'CrystalSymmetry',34,'SampleFrame',[63 15 148]*pi/180);
cases{end+1} = mk('sites_all',Sys,Exp,struct()); cases{end+1} = mk('sites_1',Sys,Exp,struct('Sites',1)); cases{end+1} = mk('sites_23',Sys,Exp,struct('Sites',[2 3]));
% crystal_th (point group Th)
Sys = struct('g',[2.0 2.1 2.2],'gFrame',[-78 -40 -30]*pi/180,'lw',0.5); [phi,theta] = vec2ang([1;2;3]); Exp = struct('mwFreq',9.5,'Range',[290 350],'SampleFrame',[0 -theta -phi],'CrystalSymmetry','Th');
cases{end+1} = mk('th',Sys,Exp,struct());
% crystal_twocrystals (two sample orientations, sum and separate)
Sys = struct('g',[2.0 2.1 2.2],'gFrame',[30 40 78]*pi/180,'lw',0.5); Exp = struct('SampleFrame',[10 24 61; 222 55 99]*pi/180,'CrystalSymmetry',130,'mwFreq',9.5,'Range',[290 350]);
cases{end+1} = mk('twocrystals_sum',Sys,Exp,struct());
Opt = struct('separate','orientations'); [B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','twocrystals_separate','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
% samplerotation: rotateframe equivalence, 1H with tilted A
Sys = struct('g',[2 2.1 2.3],'lwpp',0.2,'A',[10 50 200],'AFrame',[0 pi/5 0],'Nucs','1H'); Exp = struct('mwFreq',9.5,'Harmonic',0,'Range',[280 360]);
nrot_L = [0.2 -0.8 0.5]; rho = deg2rad(266); sf0 = [10 50 -30]*pi/180;
Exp.SampleFrame = rotateframe(sf0,nrot_L,rho); cases{end+1} = mk('samplerotation_rotated',Sys,Exp,struct());
Exp.SampleFrame = sf0; Exp.SampleRotation = {nrot_L,rho}; cases{end+1} = mk('samplerotation_viaRotation',Sys,Exp,struct());
% intensity_crystal_mx: MolFrame only, matrix method; integral vs formula
Sys = struct('g',[2.1 2.1 2.0],'lwpp',1); Exp = struct('mwFreq',9.6,'Range',[320 350],'Harmonic',0,'nPoints',10000,'MolFrame',[1.1 2.2 0.4]);
cases{end+1} = mk('intensity_crystal_mx',Sys,Exp,struct('Method','matrix'));
% intensity_isopowder_crystal: isotropic g, crystal vs powder integrals equal
Sys = struct('g',2,'lwpp',0.5); Exp = struct('mwFreq',9.6,'Range',[341 345],'Harmonic',0);
cases{end+1} = mk('isopowder_powder',Sys,Exp,struct()); Exp.SampleFrame = [1.3 2.1 4.0]; cases{end+1} = mk('isopowder_crystal',Sys,Exp,struct());
% smallgdiff_crystal (coarse and fine grids)
for nP = [100 20000]
  Exp = struct('mwFreq',9.5,'Range',[320 360],'Harmonic',0,'SampleFrame',[0 0 0],'nPoints',nP);
  cases{end+1} = mk(sprintf('smallgdiff_crystal_A_n%d',nP),struct('g',2.0001,'lw',[1 1]),Exp,struct());
  cases{end+1} = mk(sprintf('smallgdiff_crystal_B_n%d',nP),struct('g',2.0000,'lw',[1 1]),Exp,struct());
end
% strain in a crystal (HStrain) and parallel mode
Sys = struct('g',[2 2.1 2.2],'HStrain',[30 60 90]); Exp = struct('mwFreq',9.5,'Range',[290 350],'SampleFrame',[0.4 1.1 2.0],'CrystalSymmetry','D2h');
cases{end+1} = mk('crystal_hstrain',Sys,Exp,struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_crystal.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
