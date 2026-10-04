% pepper features (PORT_SPEC_2 items 3.4-3.6): non-equilibrium populations (Sys.initState),
% partial ordering (Exp.Ordering), photoselection (Exp.lightBeam/lightScatter, Sys.tdm).
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [B,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
end
function c = mkinit(name,Sys,Exp,Opt,pops,basis)
  Sys.initState = {pops,basis}; c = mk(name,Sys,Exp,Opt);
  c.Sys = rmfield(c.Sys,'initState'); c.initPops = pops; c.initBasis = basis;
end
% --- non-equilibrium populations ---
Sys = struct('S',1,'g',2,'lw',0.3,'D',[300 10]); Exp = struct('mwFreq',9.5,'Range',[325 355],'Harmonic',0);
cases{end+1} = mkinit('noneq_zerofield',Sys,Exp,struct(),[0.85 1 0.95],'zerofield');
Sysf = Sys; Sysf.lw = unitconvert(Sys.lw,'mT->MHz');
cases{end+1} = mkinit('noneq_zerofield_freq',Sysf,struct('Field',340,'Harmonic',0),struct(),[0.85 1 0.95],'zerofield');
Sys = struct('S',1,'g',2,'lw',0.3,'D',[300 30]);
cases{end+1} = mkinit('noneq_eigen',Sys,Exp,struct(),[0 1 0],'eigen');
Sys = struct('S',1,'g',2,'lw',0.3,'D',[1000 -250]); Exp = struct('mwFreq',9.5,'Range',[290 390],'Harmonic',0);
cases{end+1} = mkinit('noneq_ISCtriplet',Sys,Exp,struct(),[0.1 0.5 0.4],'zerofield');
Sysf = Sys; Sysf.lw = unitconvert(Sys.lw,'mT->MHz');
cases{end+1} = mkinit('noneq_ISCtriplet_freq',Sysf,struct('Field',340,'mwRange',[8 11],'Harmonic',0),struct(),[0.1 0.5 0.4],'zerofield');
% xyz basis with DFrame (pepper_noneqpop_ISC_xyzpops)
Tr = struct('S',1,'lwpp',1,'DFrame',[0.3 1.1 0.7],'D',[900 280]); Exp = struct('mwFreq',9.5,'Range',[290 390],'Harmonic',0);
cases{end+1} = mkinit('noneq_xyz',Tr,Exp,struct(),[0.2 0.5 0.3],'xyz');
Tr.D = [-900 280];
cases{end+1} = mkinit('noneq_xyz_negD',Tr,Exp,struct(),[0.2 0.5 0.3],'xyz');
% coupled basis SCRP with 14N (pepper_noneqpop_scrp_input)
Sys = struct('S',[1/2 1/2],'g',[2.0027; 2.0000],'J',-6,'Nucs','14N','A',[0 0 0 5 5 20],'lwpp',0.1);
Exp = struct('mwFreq',9.75,'Range',[346 350],'Harmonic',0);
cases{end+1} = mkinit('noneq_scrp_coupled',Sys,Exp,struct(),[1/3 1/3 1/3 0],'coupled');
V = cgmatrix(1/2,1/2); Tp = V(1,:)'; T0 = V(2,:)'; Tm = V(3,:)'; rho = 1/3*(Tp*Tp' + T0*T0' + Tm*Tm');
Sys.initState = rho; c = mk('noneq_scrp_densmat',Sys,Exp,struct()); c.Sys = rmfield(c.Sys,'initState'); c.initDens = rho; cases{end+1} = c;
% uncoupled density matrix, triplet with nucleus (pepper_noneqpop_ISC_couplednuclearspin-like)
Sys = struct('S',1,'g',2,'lw',0.3,'D',[300 10],'Nucs','1H','A',[10 10 20]);
Exp = struct('mwFreq',9.5,'Range',[325 355],'Harmonic',0);
cases{end+1} = mkinit('noneq_zerofield_1H',Sys,Exp,struct(),[0.85 1 0.95],'zerofield');
% temperature for reference (Boltzmann path)
Sys = struct('S',1,'g',2,'lw',0.3,'D',[1000 -250]); Exp = struct('mwFreq',9.5,'Range',[290 390],'Harmonic',0,'Temperature',2);
cases{end+1} = mk('boltzmann_2K',Sys,Exp,struct());
% --- partial ordering ---
Sys = struct('g',[2 2 2.2],'lw',1); Exp = struct('mwFreq',9.5,'Harmonic',0,'Range',[300 350]);
for lambda = [-5 -2 2 5]
  Exp.Ordering = lambda; cases{end+1} = mk(sprintf('ordering_axial_%+d',lambda),Sys,Exp,struct());
end
Sys = struct('g',[2 2.05 2.2],'lw',1); Exp = struct('mwFreq',9.5,'Harmonic',0,'Range',[300 350]); Opt = struct('GridSymmetry','Ci');
Exp.Ordering = 3; cases{end+1} = mk('ordering_rhombic_Ci_3',Sys,Exp,Opt);
Exp.Ordering = @(a,b,c) exp(-2*cos(b).^2 + 0.5*cos(2*c)); c = mk('ordering_userfun_abc',Sys,Exp,Opt); c.Exp.Ordering = 'exp(-2*cos(b).^2 + 0.5*cos(2*c))'; cases{end+1} = c;
Exp.Ordering = @(b) exp(1.5*cos(b).^2); c = mk('ordering_userfun_b',Sys,Exp,Opt); c.Exp.Ordering = 'exp(1.5*cos(b).^2)'; cases{end+1} = c;
% ordering + sample tilt (pepper_ordering_tilts)
Sys = struct('S',1/2,'lw',1,'g',[2 2.1 2.2],'gFrame',deg2rad([10 40 20]));
Exp = struct('Ordering',3,'mwFreq',9.5,'Harmonic',0,'Range',[300 350]); Opt = struct('GridSize',11,'GridSymmetry','Ci');
n = [1 4 7]; rho = deg2rad(37); Exp.SampleFrame = eulang(rotaxi2mat(n,rho));
cases{end+1} = mk('ordering_tilt_sampleframe',Sys,Exp,Opt);
% --- photoselection ---
Tr = struct('S',1,'D',[900 160],'lwpp',1,'tdm','y'); Tr.initState = {[1 0 1],'zerofield'};
Exp = struct('mwFreq',9.5,'Range',[280 400],'Harmonic',0,'lightScatter',0.2); Opt = struct('GridSymmetry','D2h');
for lb = {'','perpendicular','parallel','unpolarized'}
  Exp.lightBeam = lb{1}; c = mk(sprintf('photosel_%s',lb{1}),Tr,Exp,Opt);
  if isempty(lb{1}), c.name = 'photosel_none'; end
  c.Sys = rmfield(c.Sys,'initState'); c.initPops = [1 0 1]; c.initBasis = 'zerofield'; cases{end+1} = c;
end
Tr = struct('S',1,'D',[-1000 150],'lwpp',1,'tdm',[0 45]*pi/180); Tr.initState = {[0.2 0.2 0.6],'zerofield'};
Exp = struct('mwFreq',9.7,'Range',[280 400],'Harmonic',0,'lightScatter',0);
for lb = {'perpendicular','parallel'}
  Exp.lightBeam = lb{1}; c = mk(sprintf('photosel_tdmangles_%s',lb{1}),Tr,Exp,struct());
  c.Sys = rmfield(c.Sys,'initState'); c.initPops = [0.2 0.2 0.6]; c.initBasis = 'zerofield'; cases{end+1} = c;
end
% custom beam {k alpha} and Boltzmann (no initState)
Tr = struct('S',1,'D',[900 160],'lwpp',1,'tdm','xz'); Exp = struct('mwFreq',9.5,'Range',[280 400],'Harmonic',0,'lightScatter',0.1);
Exp.lightBeam = {[1;1;0]/sqrt(2), 0.3}; c = mk('photosel_custom_beam',Tr,Exp,struct()); c.Exp.lightBeam_k = [1 1 0]/sqrt(2); c.Exp.lightBeam_alpha = 0.3; c.Exp = rmfield(c.Exp,'lightBeam'); cases{end+1} = c;
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_features2.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
