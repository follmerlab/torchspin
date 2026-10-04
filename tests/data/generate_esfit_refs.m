% EasySpin esfit oracle: deterministic fits (levmar, simplex) of noisy synthetic spectra
% from offset starting values.  Saved: data, starting Sys, Vary, fitted parameters, rmsd, fit.
clear all; cases = {};
function c = run_case(name, Sys0, Vary, Exp, spc, method)
  FitOpt = struct('Verbosity',0,'Method',method);
  result = esfit(spc,@pepper,{Sys0,Exp},{Vary},FitOpt);
  c = struct('name',name,'Sys0',Sys0,'Vary',Vary,'Exp',Exp,'spc',spc,'method',method, ...
             'pfit',result.pfit(:)','pnames',{result.pnames},'rmsd',result.rmsd,'fit',result.fit(:)','argsfit',result.argsfit{1});
  fprintf('%s: rmsd %.4g, pfit %s\n', name, result.rmsd, mat2str(result.pfit(:)',6));
end
% 1: esfit_basic-like, frequency sweep, two g values, offset start
Sys = struct('g',[2 2.1],'lw',10); Exp = struct('Field',350,'mwRange',[9.5 10.5]);
[nu,spc] = pepper(Sys,Exp); rng(1); spc = addnoise(spc,50,'n');
Sys0 = Sys; Sys0.g = [1.99 2.115]; Vary = struct('g',[0.03 0.03]);
cases{end+1} = run_case('freq_g2_levmar',Sys0,Vary,Exp,spc,'levmar fcn');
cases{end+1} = run_case('freq_g2_simplex',Sys0,Vary,Exp,spc,'simplex fcn');
% 2: field sweep, rhombic g + lwpp
Sys = struct('g',[2.009 2.006 2.002],'lwpp',0.5); Exp = struct('mwFreq',9.5,'Range',[330 350],'nPoints',256);
[B,spc] = pepper(Sys,Exp); rng(42); spc = addnoise(spc,50,'n');
Sys0 = Sys; Sys0.g = [2.007 2.007 2.0035]; Sys0.lwpp = 0.7; Vary = struct('g',[0.005 0.005 0.005],'lwpp',0.4);
cases{end+1} = run_case('field_g3lw_levmar',Sys0,Vary,Exp,spc,'levmar fcn');
cases{end+1} = run_case('field_g3lw_simplex',Sys0,Vary,Exp,spc,'simplex fcn');
% 3: g + 14N hyperfine (A anisotropic), levmar, integrated target
Sys = struct('g',[2.008 2.006 2.003],'Nucs','14N','A',[20 20 85],'lwpp',0.3); Exp = struct('mwFreq',9.5,'Range',[330 350],'nPoints',512);
[B,spc] = pepper(Sys,Exp); rng(43); spc = addnoise(spc,80,'n');
Sys0 = Sys; Sys0.g = [2.0075 2.0065 2.0025]; Sys0.A = [18 22 82]; Vary = struct('g',[0.003 0.003 0.003],'A',[6 6 8]);
cases{end+1} = run_case('field_gA_levmar',Sys0,Vary,Exp,spc,'levmar fcn');
cases{end+1} = run_case('field_gA_levmar_int',Sys0,Vary,Exp,spc,'levmar int');
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_esfit.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
