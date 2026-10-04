% MATLAB/EasySpin counterparts of benchmarks/workloads/workloads.py (same parameters).
% Run:  matlab -batch "addpath(genpath('<EasySpin>/easyspin')); cd benchmarks/workloads; matlab_workloads"
% Writes matlab_workloads.json next to this file: seconds (median per workload),
% replicates (every timed call), and meta (MATLAB version, thread counts, affinity).
% Environment: BENCH_WARMUP untimed calls (default 0) then BENCH_NREP timed
% calls (default 1) per workload; BENCH_OUT overrides the output file name.
clear all; res = struct(); reps = struct(); Opt0 = struct('Verbosity',0);
nWarm = str2double(getenv('BENCH_WARMUP')); if isnan(nWarm), nWarm = 0; end
nRep = str2double(getenv('BENCH_NREP')); if isnan(nRep), nRep = 1; end
function [t, r] = tm(f, nWarm, nRep)
  for k = 1:nWarm, f(); end
  r = zeros(1, nRep);
  for k = 1:nRep, tic; f(); r(k) = toc; end
  t = median(r);
end
function rep(name, t, r)
  if numel(r) > 1
    q = interp1(linspace(0, 1, numel(r)), sort(r), [0.25 0.75]);   % no toolbox dependency
    fprintf('%-32s %9.2f s  (median of %d, IQR %.2f-%.2f)\n', name, t, numel(r), q(1), q(2));
  else
    fprintf('%-32s %9.2f s\n', name, t);
  end
end
% pepper 63Cu + 2x14N, matrix, 19x4
Sys = struct('S',1/2,'g',[2.05 2.05 2.25],'Nucs','63Cu,14N,14N','A',[30 30 500; 40 40 40; 40 40 40],'lwpp',0.5);
Exp = struct('mwFreq',9.5,'Range',[260 360],'nPoints',2048);
[res.pepper_cu_2n_matrix, reps.pepper_cu_2n_matrix] = tm(@() pepper(Sys,Exp,setfield(Opt0,'Method','matrix')), nWarm, nRep); rep('pepper_cu_2n_matrix', res.pepper_cu_2n_matrix, reps.pepper_cu_2n_matrix);
% pepper Mn(II) S=5/2 + 55Mn, matrix
Sys = struct('S',5/2,'g',2,'D',[600 100],'Nucs','55Mn','A',250,'lwpp',1);
[res.pepper_mn_S52_matrix, reps.pepper_mn_S52_matrix] = tm(@() pepper(Sys,Exp,setfield(Opt0,'Method','matrix')), nWarm, nRep); rep('pepper_mn_S52_matrix', res.pepper_mn_S52_matrix, reps.pepper_mn_S52_matrix);
% pepper perturb2, dense grid 91
Sys = struct('S',1/2,'g',[2.0023 2.0025 2.0035],'Nucs','14N,1H,1H','A',[20 20 90; 8 8 12; 5 5 7],'lwpp',0.2);
Exp = struct('mwFreq',9.5,'Range',[330 350],'nPoints',4096);
[res.pepper_perturb_dense_grid, reps.pepper_perturb_dense_grid] = tm(@() pepper(Sys,Exp,struct('Verbosity',0,'Method','perturb2','GridSize',[91 1])), nWarm, nRep); rep('pepper_perturb_dense_grid', res.pepper_perturb_dense_grid, reps.pepper_perturb_dense_grid);
% pepper strain summation 61x4
Sys = struct('S',1/2,'g',[2.008 2.006 2.003],'Nucs','14N','A',[20 20 85],'gStrain',[0.003 0.002 0.001],'HStrain',[10 10 30],'lwpp',0.3);
Exp = struct('mwFreq',9.5,'Range',[330 350],'nPoints',2048);
[res.pepper_strain_summation, reps.pepper_strain_summation] = tm(@() pepper(Sys,Exp,struct('Verbosity',0,'GridSize',[61 4])), nWarm, nRep); rep('pepper_strain_summation', res.pepper_strain_summation, reps.pepper_strain_summation);
% fit loop: 20 forward calls
Exp = struct('mwFreq',9.5,'Range',[330 350],'nPoints',1024); gs = linspace(2.000,2.010,20);
function loop20(gs,Exp)
  for k = 1:numel(gs)
    Sys = struct('S',1/2,'g',[gs(k) 2.006 2.003],'Nucs','14N','A',[20 20 85],'lwpp',0.3);
    pepper(Sys,Exp,struct('Verbosity',0,'GridSize',[31 4]));
  end
end
[res.pepper_fit_loop_20, reps.pepper_fit_loop_20] = tm(@() loop20(gs,Exp), nWarm, nRep); rep('pepper_fit_loop_20', res.pepper_fit_loop_20, reps.pepper_fit_loop_20);
% chili nitroxide + 1H, tcorr 30 ns
Sys = struct('S',1/2,'g',[2.008 2.006 2.003],'Nucs','14N,1H','A',[20 20 85; 5 5 8],'tcorr',3e-8,'lw',[0.1 0.1]);
Exp = struct('mwFreq',9.5,'Range',[330 350],'nPoints',1024);
[res.chili_nitroxide_2nuc, reps.chili_nitroxide_2nuc] = tm(@() chili(Sys,Exp,Opt0), nWarm, nRep); rep('chili_nitroxide_2nuc', res.chili_nitroxide_2nuc, reps.chili_nitroxide_2nuc);
% chili powder with potential
Sys = struct('S',1/2,'g',[2.008 2.006 2.003],'Nucs','14N','A',[20 20 85],'tcorr',2e-8,'lw',[0.1 0.1],'Potential',[2 0 0 1.5]);
% EasySpin requires an explicit basis with Sys.Potential; [14 7 2 6] is the default torchspin (and EasySpin) LLMK
[res.chili_powder_potential, reps.chili_powder_potential] = tm(@() chili(Sys,Exp,struct('Verbosity',0,'LLMK',[14 7 2 6])), nWarm, nRep); rep('chili_powder_potential', res.chili_powder_potential, reps.chili_powder_potential);
% saffron HYSCORE 512x512, GridSize 91
Sys = struct('S',1/2,'g',[2.0023 2.0023 2.0023],'Nucs','14N,1H','A',[3 3 6; 2 2 8],'Q',[-0.5 -0.5 1.0; 0 0 0]);
Exp = struct('Sequence','HYSCORE','Field',350,'dt',0.016,'nPoints',512,'tau',0.1);
[res.saffron_hyscore_512, reps.saffron_hyscore_512] = tm(@() saffron(Sys,Exp,struct('Verbosity',0,'GridSize',91)), nWarm, nRep); rep('saffron_hyscore_512', res.saffron_hyscore_512, reps.saffron_hyscore_512);
% cardamom diffusion/fast 200 x 1000
Sys = struct('S',1/2,'g',[2.008 2.006 2.003],'Nucs','14N','A',[20 20 85],'tcorr',1e-9);
Exp = struct('mwFreq',9.5,'Range',[332 352],'nPoints',256,'Harmonic',0);
Par = struct('Model','diffusion','nTraj',200,'nSteps',1000,'dtSpin',1e-10,'dtSpatial',1e-10);
[res.cardamom_diffusion_200x1000, reps.cardamom_diffusion_200x1000] = tm(@() cardamom(Sys,Exp,Par,struct('Method','fast','Verbosity',0)), nWarm, nRep); rep('cardamom_diffusion_200x1000', res.cardamom_diffusion_200x1000, reps.cardamom_diffusion_200x1000);
outName = getenv('BENCH_OUT'); if isempty(outName), outName = 'matlab_workloads.json'; end
meta = struct('matlab', version, 'host', getenv('HOSTNAME'), 'maxNumCompThreads', maxNumCompThreads, ...
              'numcores', feature('numcores'), 'warmup', nWarm, 'repeat', nRep, ...
              'protocol', 'median of replicates; every replicate saved', 'started', string(datetime('now')));
try, meta.easyspin_path = fileparts(which('pepper')); catch, end
try, [~, aff] = system('taskset -pc $PPID'); meta.cpu_affinity = strtrim(aff); catch, end
fid = fopen(fullfile(fileparts(mfilename('fullpath')), outName), 'w');
fprintf(fid, '%s\n', jsonencode(struct('meta', meta, 'seconds', res, 'replicates', reps))); fclose(fid);
