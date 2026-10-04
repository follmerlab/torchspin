% PORT_SPEC_2 item 1.4: pepper MATLAB references mirroring EasySpin's own
% pepper_*.m regression/consistency tests (fixed values replace rand).
% Saved to ref_pepper_ext2.mat (-v7) as a cell array `cases`; each case stores
% name, Sys, Exp, Opt and the spectrum (B, spc). Scalar Opt.GridSize follows
% EasySpin semantics (N == [N 4]).
clear all
fprintf('Generating pepper ext2 references...\n');
cases = {};

% pepper_axiallw / pepper_rhombiclw / pepper_s_optional
cases{end+1} = mk('axiallw', struct('S',1/2,'g',[2 2 2.2],'lw',1), struct('mwFreq',9.5,'Range',[300 350]), struct());
cases{end+1} = mk('rhombiclw', struct('S',1/2,'g',[2 2.1 2.2],'lw',1), struct('mwFreq',9.5,'Range',[300 350]), struct());
cases{end+1} = mk('s_optional', struct('g',[2 2 2.2],'lw',3), struct('mwFreq',9.5,'Range',[280 350]), struct());

% pepper_fieldrange (Q band, autorange -> explicit Range)
Sys = struct('g',[2 2.1 2.2],'lw',1); Exp = struct('mwFreq',35);
[x0,~] = pepper(Sys,Exp); Exp.Range = [min(x0) max(x0)];
cases{end+1} = mk('fieldrange_Q', Sys, Exp, struct());

% pepper_gausslorentz: harmonics 0..2 x lineshape variants
Sys = struct('S',1/2,'g',2); Exp = struct('mwFreq',9.5,'Range',[335 344]);
variants = {'G',1; 'G0',[1 0]; 'GL',[1 1]; 'L',[0 1]};
for H = 0:2
  Exp.Harmonic = H;
  for v = 1:size(variants,1)
    Sys.lw = variants{v,2};
    cases{end+1} = mk(sprintf('gausslorentz_h%d_%s',H,variants{v,1}), Sys, Exp, struct());
  end
end

% pepper_harmonic (gStrain, perturb1) and pepper_harmonic_iso (axial g, lw)
Sys = struct('S',1/2,'g',[2 2.15 2.3],'gStrain',[0.03 0.005 0.02]);
Exp = struct('mwFreq',9.7979,'Range',[290 360],'nPoints',10000);
Opt = struct('Verbosity',0,'GridSize',[30 1],'Method','perturb1');
for H = 0:2, Exp.Harmonic = H; cases{end+1} = mk(sprintf('harmonic_gstrain_h%d',H), Sys, Exp, Opt); end
Sys = struct('S',1/2,'g',[2 2 2.3],'lw',2); Opt = struct('Verbosity',0,'GridSize',[30 1]);
for H = 0:2, Exp.Harmonic = H; cases{end+1} = mk(sprintf('harmonic_iso_h%d',H), Sys, Exp, Opt); end

% pepper_lwpplw
Sys = struct('g',2,'Nucs','1H','A',10,'lw',[0.1 0.1]);
cases{end+1} = mk('lwpplw', Sys, struct('mwFreq',9.7,'CenterSweep',[346.5 1]), struct());

% pepper_nobroadening (no lw: EasySpin auto-selects Harmonic 0)
Sys = struct('Nucs','14N','A',[20 20 100]); Exp = struct('mwFreq',0.250,'Range',[1 15]);
cases{end+1} = mk('nobroadening_auto', Sys, Exp, struct());
Exp.Harmonic = 0; cases{end+1} = mk('nobroadening_h0', Sys, Exp, struct());

% pepper_relativebroad: HStrain series, perturb, GridSize [19 3]
Sys = struct('S',1/2,'g',[2 2.4 3]); Exp = struct('mwFreq',9.5,'Range',[200 400],'Harmonic',0,'nPoints',1e4);
Opt = struct('Verbosity',0,'GridSize',[19 3],'Method','perturb');
for lw = [1 3 10 30 100 300]
  Sys.HStrain = [1 1 1]*lw; cases{end+1} = mk(sprintf('relativebroad_hs%d',lw), Sys, Exp, Opt);
end

% pepper_smallgdiff_isopowder / _lorentzian
sgv = {'isopowder',[1 1]; 'lorentzian',[0 2]};
for iv = 1:2
  for nP = [100 20000]
    Exp = struct('mwFreq',9.5,'Range',[320 360],'Harmonic',0,'nPoints',nP);
    cases{end+1} = mk(sprintf('smallgdiff_%s_A_n%d',sgv{iv,1},nP), struct('g',2.0001,'lw',sgv{iv,2}), Exp, struct());
    cases{end+1} = mk(sprintf('smallgdiff_%s_B_n%d',sgv{iv,1},nP), struct('g',2.0000,'lw',sgv{iv,2}), Exp, struct());
  end
end

% pepper_interpolation
Sys = struct('S',.5,'g',[1.9 2.01 2.3],'lw',1); Exp = struct('Range',[285 365],'mwFreq',9.5);
cases{end+1} = mk('interpolation_D2h_19x5', Sys, Exp, struct('GridSymmetry','D2h','GridSize',[19 5]));
cases{end+1} = mk('interpolation_D2h_91x1', Sys, Exp, struct('GridSymmetry','D2h','GridSize',[91 1]));

% pepper_fullsphere
Sys = struct('g',[2 2.05 2.2],'lw',1); Exp = struct('mwFreq',9.5,'Harmonic',0,'Range',[305 345]);
for sym = {'D2h','Ci','C1'}
  cases{end+1} = mk(['fullsphere_' sym{1}], Sys, Exp, struct('GridSize',[10 2],'GridSymmetry',sym{1}));
end

% pepper_c2h
Sys = struct('g',[1.9 2 2.3],'Nucs','1H','A',[20 80 300],'AFrame',[-pi/4 -pi/2 0],'lw',1);
Exp = struct('Range',[285 370],'mwFreq',9.5,'Harmonic',0);
cases{end+1} = mk('c2h_Ci', Sys, Exp, struct('GridSize',[50 3],'Method','perturb','GridSymmetry','Ci'));
cases{end+1} = mk('c2h_C2h', Sys, Exp, struct('GridSize',[50 3],'Method','perturb','GridSymmetry','C2h'));

% pepper_eig_matrix (matrix branch only)
cases{end+1} = mk('eig_matrix', struct('g',2.1,'Nucs','1H','A',100,'lwpp',1), struct('mwFreq',9.5,'Range',[300 360],'Harmonic',0), struct('Method','matrix'));

% pepper_perturb_fullg
Sys = struct('g',[2 0 0; 0 2.05 0; 0 0 2.1],'lwpp',1); Exp = struct('mwFreq',9.7,'CenterSweep',[338 40]);
for m = {'matrix','perturb1','perturb2'}, cases{end+1} = mk(['perturb_fullg_' m{1}], Sys, Exp, struct('Method',m{1})); end

% pepper_perturb_matrix_highspin
Sys = struct('S',3/2,'D',300,'g',[2 2.2],'lw',1); Exp = struct('mwFreq',9.5,'Range',[250 380],'Harmonic',0);
for m = {'matrix','perturb2'}, cases{end+1} = mk(['perturb_highspin_' m{1}], Sys, Exp, struct('GridSize',91,'Method',m{1})); end

% pepper_perturb_matrix_orthog
Sys = struct('g',[2 4 6],'lw',5); Exp = struct('mwFreq',9.5,'Range',[80 380],'Harmonic',1);
for m = {'matrix','perturb2'}, cases{end+1} = mk(['perturb_orthog_' m{1}], Sys, Exp, struct('Method',m{1})); end

% pepper_fulld / pepper_dinput
Exp = struct('mwFreq',9.5,'Range',[200 400]);
cases{end+1} = mk('fulld_matrix', struct('S',1,'lwpp',1,'D',diag([-1 -1 2]*400/3)), Exp, struct());
cases{end+1} = mk('fulld_scalar', struct('S',1,'lwpp',1,'D',400), Exp, struct());
Exp = struct('mwFreq',9.5,'Range',[333 345]);
cases{end+1} = mk('dinput_principal', struct('S',1,'D',[-1 -1 2]*99/3+[1 -1 0]*10,'lw',0.1), Exp, struct());
cases{end+1} = mk('dinput_DE', struct('S',1,'D',[99 10],'lw',0.1), Exp, struct());

% pepper_eespins
Sys = struct('S',[1/2 1/2],'g',[2 2.05 2.1; 2.2 2.25 2.3],'HStrain',[1 1 1]*30); Sys.ee = [1 1 -2]*100;
cases{end+1} = mk('eespins', Sys, struct('mwFreq',9.5,'Range',[280 350]), struct());

% pepper_twocomponents
Sys1 = struct('g',[2 2.2],'lwpp',1,'weight',0.567); Sys2 = struct('g',[2.05 2.1 2.15],'lwpp',2,'weight',1.734);
Exp = struct('mwFreq',9.7,'Range',[300 360]);
cases{end+1} = mk('twocomponents_1', Sys1, Exp, struct());
cases{end+1} = mk('twocomponents_2', Sys2, Exp, struct());
[B,spc] = pepper({Sys1,Sys2},Exp);
cases{end+1} = struct('name','twocomponents_sum','Sys',Sys1,'Sys2',Sys2,'Exp',Exp,'Opt',struct(),'B',B,'spc',spc);

% pepper_temperature
Sys = struct('S',1,'g',[1 1 1]*2,'D',200*[1 1 -2],'lw',1); Exp = struct('Range',[300 380],'mwFreq',9.5,'Harmonic',0);
for T = [20 10 5 2 1 0.5 0.2]
  Exp.Temperature = T; cases{end+1} = mk(sprintf('temperature_%gK',T), Sys, Exp, struct('GridSize',20));
end

% pepper_pt_* (perturbation vs matrix consistency systems)
Sys = struct('Nucs','1H','g',[2 2.3 2.7],'A',[350 400],'HStrain',[20 40 70],'gStrain',[0.04 0.02],'AStrain',[90 10],'gAStrainCorr',-1);
Exp = struct('mwFreq',9.5,'Range',[200 400]);
for m = {'matrix','perturb2'}, cases{end+1} = mk(['pt_broadenings_' m{1}], Sys, Exp, struct('Method',m{1})); end
Sys = struct('S',1,'g',2,'D',600,'lwpp',1); Exp = struct('mwFreq',9.8,'CenterSweep',[350 100],'Harmonic',1);
for m = {'perturb','matrix'}, cases{end+1} = mk(['pt_donly_' m{1}], Sys, Exp, struct('Method',m{1})); end
Sys = struct('S',1,'g',2,'lwpp',1,'D',[-300 -300 600]);
cases{end+1} = mk('pt_dtrace_traceless', Sys, Exp, struct('Method','perturb'));
Sys.D = [-300 -300 600]+100; cases{end+1} = mk('pt_dtrace_shifted', Sys, Exp, struct('Method','perturb'));
Sys = struct('S',1/2,'g',[2 2.2],'Nucs','63Cu','A',[50 500],'lwpp',1); Exp = struct('mwFreq',9.8,'CenterSweep',[330 100],'Harmonic',1);
for m = {'perturb','matrix'}, cases{end+1} = mk(['pt_ga1_' m{1}], Sys, Exp, struct('Method',m{1})); end
Sys = struct('S',1/2,'g',[2 2.2],'Nucs','63Cu,1H','A',[70 500; 50 50],'lwpp',0.4);
for m = {'perturb','matrix'}, cases{end+1} = mk(['pt_ga2_' m{1}], Sys, Exp, struct('Method',m{1},'GridSize',31)); end
Sys = struct('S',1,'D',400,'g',[2 2.1 2.2],'Nucs','1H','A',[150 200],'lwpp',1); Exp = struct('mwFreq',9.8,'CenterSweep',[330 160],'Harmonic',1);
for m = {'perturb','matrix'}, cases{end+1} = mk(['pt_gda_' m{1}], Sys, Exp, struct('Method',m{1})); end
Sys = struct('S',3/2,'D',300,'g',[2 2.3],'lwpp',2);
for m = {'perturb','matrix'}, cases{end+1} = mk(['pt_gdonly_' m{1}], Sys, Exp, struct('Method',m{1},'GridSize',31)); end
Sys = struct('g',[2 2.1 2.2],'lwpp',2); Exp = struct('mwFreq',9.8,'CenterSweep',[335 60],'Harmonic',1);
for m = {'perturb','matrix'}, cases{end+1} = mk(['pt_gonly_' m{1}], Sys, Exp, struct('Method',m{1})); end
Sys = struct('g',[2.0 2.1 2.2],'HStrain',[1 1 1]*200); Exp = struct('mwFreq',35,'Range',[1000 1300]);
for m = {'matrix','perturb'}, cases{end+1} = mk(['pt_hstrain_' m{1}], Sys, Exp, struct('Method',m{1})); end
Sys = struct('S',1,'g',2,'D',400,'lwpp',0.5); Exp = struct('mwFreq',9.5,'CenterSweep',[330 60],'Temperature',0.5,'Harmonic',0);
for m = {'matrix','perturb'}, cases{end+1} = mk(['pt_temp_' m{1}], Sys, Exp, struct('Method',m{1})); end

% pepper_gstrain / pepper_gstrain2
Sys = struct('S',1/2,'g',[2 2.1 2.2],'gStrain',[1 2 3]*0.01); Exp = struct('mwFreq',9.5,'Range',[290 350]);
cases{end+1} = mk('gstrain_matrix', Sys, Exp, struct('Method','matrix'));
cases{end+1} = mk('gstrain_perturb', Sys, Exp, struct('Method','perturb'));

% pepper_convharmonics
Sys = struct('g',1.9995); Exp = struct('mwFreq',9.5,'Range',[339 340]);
cv = {'G',0.1; 'G0',[0.1 0]; 'L',[0 0.08]; 'GL',[0.1 0.08]};
for v = 1:4, Sys.lwpp = cv{v,2}; cases{end+1} = mk(['convharmonics_' cv{v,1}], Sys, Exp, struct()); end

% pepper_dispersion_fieldsweep (absorption and dispersion via mwPhase)
gam = gammae; T2 = 1e-7; om = 2*pi*9.525e9; B0 = 0.340 + linspace(-1,1,1e4)*0.002;
Sys = struct('g',gfree); fwhm = 1/pi/T2; fwhm = fwhm*planck/bmagn/Sys.g/1e-3; Sys.lw = [0 fwhm];
Exp = struct('mwFreq',om/2/pi/1e9,'Range',[min(B0) max(B0)]*1e3,'nPoints',numel(B0),'Harmonic',0);
cases{end+1} = mk('dispersion_absorption', Sys, Exp, struct());
Exp.mwPhase = pi/2; cases{end+1} = mk('dispersion_dispersion', Sys, Exp, struct());

% pepper_freqsweep_basic / pepper_freqsweep_gstrain
cases{end+1} = mk('freqsweep_basic', struct('g',[2 2.05 2.01],'lwpp',10), struct('Field',340,'mwRange',[9 10]), struct());
cases{end+1} = mk('freqsweep_gstrain', struct('g',[2.05 2 1.95],'gStrain',[0.02 0.01 0.003]), struct('Field',340,'mwRange',[9 10]), struct());

% pepper_isotropicpowder (summed spectrum)
Sys = struct('S',1/2,'g',[2 2 2],'Nucs','63Cu','A',[40 40 40],'HStrain',[1 1 1]*10);
cases{end+1} = mk('isotropicpowder', Sys, struct('mwFreq',9.7979,'Range',[340 360],'nPoints',10000), struct('Verbosity',0,'GridSize',[30 1]));

% pepper_isopowder (frequency sweep of a coupled pair, two grid symmetries)
Sys = struct('S',[1/2 1/2],'g',[2 2.15],'ee',400,'lw',30);
Exp = struct('Field',350,'Harmonic',0,'mwRange',[9.2 11],'nPoints',50000);
cases{end+1} = mk('isopowder_Dinfh', Sys, Exp, struct('GridSymmetry','Dinfh'));
cases{end+1} = mk('isopowder_Ci', Sys, Exp, struct('GridSymmetry','Ci'));

script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_ext2.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));

function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0;
  [B,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
end
