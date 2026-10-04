% chili MATLAB references (PORT_SPEC_2 Phase 2), mirroring EasySpin's chili_* tests.
% Every case stores Sys/Exp/Opt and the spectrum (x,y). Autoranged cases store
% the resulting x so the Python test can pass the same Range.
clear all
fprintf('Generating chili references...\n');
cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0;
  [x,y] = chili(Sys,Exp,Opt);
  if ~isfield(Exp,'Range') && ~isfield(Exp,'mwRange') && isfield(Exp,'mwFreq'), Exp.Range = [min(x) max(x)]; end
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'y',y);
end
% chili_simple: tcorr series
Sys = struct('g',[2.008 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'lw',0.1); Exp = struct('mwFreq',9.8,'Range',[343 358]);
for tc = [1e-9 3.16e-9 1e-8 3.16e-8 1e-7], Sys.tcorr = tc; cases{end+1} = mk(sprintf('simple_tc%g',tc),Sys,Exp,struct()); end
% chili_general_fieldsweeps
Sys = struct('g',[2.01 2.003],'lw',0.1,'tcorr',1e-9); Exp = struct('mwFreq',9.5,'Harmonic',0,'nPoints',200,'Range',[337 339]);
cases{end+1} = mk('fieldsweep_approxlin',Sys,Exp,struct('LiouvMethod','general','FieldSweepMethod','approxlin'));
cases{end+1} = mk('fieldsweep_explicit',Sys,Exp,struct('LiouvMethod','general','FieldSweepMethod','explicit'));
cases{end+1} = mk('fieldsweep_approxinv',Sys,Exp,struct('LiouvMethod','general','FieldSweepMethod','approxinv'));
% chili_nucspins (frequency sweeps)
Exp = struct('Field',350,'mwRange',[9.4 10.2],'nPoints',3000); Opt = struct('LLMK',[30 0 0 0]); A = [200 400]*2;
Sys = struct('g',[2 2.001],'tcorr',1e-6,'lw',5);
Sys.Nucs = '15N'; Sys.A = A/2; cases{end+1} = mk('nucspins_15N',Sys,Exp,Opt);
Sys.Nucs = '14N'; Sys.A = A/3; cases{end+1} = mk('nucspins_14N',Sys,Exp,Opt);
Sys.Nucs = '63Cu'; Sys.A = A/4; cases{end+1} = mk('nucspins_63Cu',Sys,Exp,Opt);
% chili_simplepotential
Nx = struct('Nucs','14N','g',[2.009 2.006 2.002],'A',unitconvert([5 5.5 33]/10,'mT->MHz'),'logDiff',7,'lwpp',[0 0]);
Exp = struct('mwFreq',9.54445,'CenterSweep',[340 12],'nPoints',512); Opt = struct('LLMK',[10 5 2 2],'GridSize',1);
Nx.Potential = [2 0 0 +1]; cases{end+1} = mk('simplepotential',Nx,Exp,Opt);
Opt.GridSize = 19; cases{end+1} = mk('simplepotential_grid19',Nx,Exp,Opt);
% chili_rhombicdiff
Sys = struct('g',[2.1 2.0 1.9],'lw',1); Exp = struct('mwFreq',9.5,'CenterSweep',[340 100]); q = 0.3e8;
Sys.Diff = [10 1 1]*q; cases{end+1} = mk('rhombicdiff_x',Sys,Exp,struct());
Sys.Diff = [1 10 1]*q; cases{end+1} = mk('rhombicdiff_y',Sys,Exp,struct());
Sys.Diff = [1 1 10]*q; cases{end+1} = mk('rhombicdiff_z',Sys,Exp,struct());
% chili_general_appfield (S=1)
Sys = struct('S',1,'g',[2.01 2.005 2.002],'D',100,'tcorr',10e-9); Exp = struct('mwFreq',9.5,'Range',[331 346],'Harmonic',0);
cases{end+1} = mk('general_appfield_S1',Sys,Exp,struct('LiouvMethod','general'));
% chili_fast_appfield (14N, treated by general method too)
Sys = struct('S',1/2,'g',[2.01 2.005 2.002],'Nucs','14N','A',[20 20 100],'tcorr',10e-9); Exp = struct('mwFreq',9.5,'Range',[332 346],'Harmonic',0);
cases{end+1} = mk('appfield_14N',Sys,Exp,struct());
% chili_fullA / fullg
g = [2.008 2.006 2.002]; A = [20 20 85]; Sys = struct('Nucs','15N','tcorr',4e-9); Exp = struct('mwFreq',9.8,'Range',[344 354]);
Sys.g = g; Sys.A = A; cases{end+1} = mk('fullA_principal',Sys,Exp,struct());
Sys.g = diag(g); Sys.A = diag(A); cases{end+1} = mk('fullA_fullboth',Sys,Exp,struct());
Sys = struct('g',[2 2.005 2.02],'tcorr',4e-9); cases{end+1} = mk('fullg_principal',Sys,Exp,struct());
Sys.g = diag([2 2.005 2.02]); cases{end+1} = mk('fullg_full',Sys,Exp,struct());
% chili_fullA_twonuclei
Sys = struct('Nucs','15N,1H','tcorr',4e-9,'g',g); Sys.A = [A; 5 10 20]; cases{end+1} = mk('twonuclei_15N_1H',Sys,Exp,struct('LLMK',[8 0 2 2]));
% chili_postconvolution
Sys = struct('g',[2.08 2.006 2.002],'Nucs','14N,1H','A',[20 20 100; 5 5 8],'tcorr',0.02e-9); Exp = struct('mwFreq',9.5,'Range',[330 338],'nPoints',1e4);
cases{end+1} = mk('postconv_full',Sys,Exp,struct('LLMK',[6 0 2 2]));
cases{end+1} = mk('postconv_pc2',Sys,Exp,struct('LLMK',[6 0 2 2],'PostConvNucs',2));
% chili_potential_general (frequency sweep, highField)
Sys = struct('g',[2.05 2.00],'logDiff',7); Exp = struct('Field',340,'mwRange',[9.4 9.9],'nPoints',200,'SampleFrame',[0 0 0]);
Opt = struct('LLMK',[4 0 2 2],'highField',true,'LiouvMethod','general'); lam = 2;
LMKlam = [2 0 0 lam; 2 1 0 lam; 2 0 1 lam; 2 1 1 lam];
for p = 1:4, Sys.Potential = LMKlam(p,:); cases{end+1} = mk(sprintf('potgeneral_L%dM%dK%d',LMKlam(p,1),LMKlam(p,2),LMKlam(p,3)),Sys,Exp,Opt); end
% chili_twomethods (general)
Sys = struct('g',[2.008 2.0061 2.0027],'Nucs','14N','A',[20 20 100],'tcorr',1e-9); Exp = struct('mwFreq',9.5,'Harmonic',0);
cases{end+1} = mk('twomethods_general',Sys,Exp,struct('LiouvMethod','general'));
% chili_freqsweep
Sys = struct('g',[2.01 2.02 2.03],'Nucs','14N','A',[10 20 30],'tcorr',1e-8,'lw',0.1); Exp = struct('Field',340,'mwRange',[9.4 9.8]);
cases{end+1} = mk('freqsweep_14N',Sys,Exp,struct());
% chili_freqderiv (S=1)
Sys = struct('S',1,'g',[2.01 2.005 2.002],'D',500,'tcorr',10e-9); Exp = struct('Field',339.4,'mwRange',[8.4 10.5],'Harmonic',1);
cases{end+1} = mk('freqderiv_S1',Sys,Exp,struct()); Sys.lwpp = 2; cases{end+1} = mk('freqderiv_S1_lwpp',Sys,Exp,struct());
% chili_frqdep
Sys = struct('g',[2.008 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'lw',0.1,'tcorr',3e-9);
for mw = [3 9 35 95], cases{end+1} = mk(sprintf('frqdep_%gGHz',mw),Sys,struct('mwFreq',mw),struct()); end
% chili_mwphase
Sys = struct('g',[2.01 2.00],'tcorr',1e-9); Exp = struct('mwFreq',9.8,'Harmonic',0,'Range',[345 353],'mwPhase',pi/2);
cases{end+1} = mk('mwphase_90',Sys,Exp,struct()); Exp.mwPhase = 0; cases{end+1} = mk('mwphase_0',Sys,Exp,struct());
% chili_temperature
Sys = struct('g',[2 2.02],'lwpp',0.1,'tcorr',20e-9); Exp = struct('mwFreq',9.5,'Range',[335 341]);
for T = [100 300 1000], Exp.Temperature = T; cases{end+1} = mk(sprintf('temperature_%dK',T),Sys,Exp,struct()); end
% chili_magnetictilt
Sys = struct('g',2.0088,'Nucs','14N','A',[17 17 90],'Diff',6e7); Exp = struct('mwFreq',9.5,'Range',[330 345]);
Sys.AFrame = [0 0 0]; cases{end+1} = mk('magnetictilt_0',Sys,Exp,struct());
Sys.AFrame = [0 30 0]*pi/180; cases{end+1} = mk('magnetictilt_30',Sys,Exp,struct());
% chili_jkmin_psmin (rhombic diffusion + potential, powder grid)
Sys = struct('g',[2.05 2.03 2.00],'tcorr',[1 2 3]*1e-9,'Potential',[2 0 0 1; 2 0 2 1; 4 0 0 1; 4 0 2 1]); Exp = struct('mwFreq',9.5,'Range',[326 344]);
for hf = [true false], for jk = [-1 1]
  cases{end+1} = mk(sprintf('jkmin%+d_highfield%d',jk,hf),Sys,Exp,struct('LLMK',[6 3 2 2],'GridSize',5,'jKmin',jk,'highField',hf));
end, end
% chili_largebasis_nan
Sys = struct('g',[2.008 2.003],'lw',0.01,'tcorr',10e-6); Exp = struct('mwFreq',9.5,'CenterSweep',[338.4 1.5]); Opt = struct('LLMK',[50 1 1 1]);
cases{end+1} = mk('largebasis_nonuc',Sys,Exp,Opt);
Sys.Nucs = '1H'; Sys.A = 10; cases{end+1} = mk('largebasis_1H',Sys,Exp,Opt);
% chili_twocomponents
Sys = struct('g',[2.008 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'lw',0.1); Exp = struct('mwFreq',9.7,'Range',[340 354]);
Sys1 = Sys; Sys1.tcorr = 0.2e-9; Sys1.weight = 0.05; Sys2 = Sys; Sys2.tcorr = 3e-9; Sys2.weight = 1;
cases{end+1} = mk('twocomp_1',Sys1,Exp,struct()); cases{end+1} = mk('twocomp_2',Sys2,Exp,struct());
[x,y] = chili({Sys1,Sys2},Exp); cases{end+1} = struct('name','twocomp_sum','Sys',Sys1,'Sys2',Sys2,'Exp',Exp,'Opt',struct(),'x',x,'y',y);
% chili_momdsimple
Nx = struct('g',[2.008 2.006 2.003],'Nucs','14N','A',[20 20 85],'lw',0.3,'tcorr',1e-8,'Potential',[2 0 0 1]);
cases{end+1} = mk('momdsimple',Nx,struct('mwFreq',9.5,'CenterSweep',[338 20]),struct('GridSize',5,'LLMK',[8 0 2 2]));
% chili_solvers (L, \, E)
Sys = struct('g',[2 2 2.1],'tcorr',1e-9); Exp = struct('Field',350,'mwRange',[9 11],'nPoints',100);
for sv = {'L','\','E'}, cases{end+1} = mk(['solver_' regexprep(sv{1},'\\','backslash')],Sys,Exp,struct('LLMK',[8 0 2 0],'Solver',sv{1})); end
% chili_rhombicg
Sys = struct('g',[2.06 2.03 2.0],'tcorr',10e-9); Exp = struct('Field',333,'mwRange',[9 10],'Harmonic',0);
cases{end+1} = mk('rhombicg',Sys,Exp,struct('LLMK',[6 0 4 4],'MpSymm',false,'highField',false,'jKmin',-1,'evenK',false,'LiouvMethod','general'));
% chili_pepper_fast_appfield (fast-motion limit vs pepper)
Sys = struct('g',[2.01 2.003],'tcorr',1e-5,'lw',0.2); Exp = struct('mwFreq',9.5,'Harmonic',0,'Range',[337 339.5]);
[xp,yp] = pepper(Sys,Exp); c = mk('pepperlimit',Sys,Exp,struct('FieldSweepMethod','approxinv','LLMK',[20 0 0 0])); c.y_pepper = yp; cases{end+1} = c;
% chili_twomethods_potential (general)
Sys = struct('g',[2 2.05 2.1],'tcorr',1e-9,'Potential',[2 0 0 1]); Exp = struct('Field',350,'mwRange',[9.4 10.6]);
cases{end+1} = mk('potential_freqsweep',Sys,Exp,struct('LLMK',[6 0 2 2],'LiouvMethod','general'));
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_chili_ext.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
