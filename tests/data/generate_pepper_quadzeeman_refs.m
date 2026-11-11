% Field-quadratic higher-order Zeeman terms (Sys.Ham2xx) in pepper, for
% torchspin Options.QuadraticZeeman (0.7.1). EasySpin's resfields/resfreqs_matrix
% build the level model from the full ham(Sys,B) at every field (gethamdata_hO),
% so the lB = 2 terms enter the resonance fields. Run from the repo root:
%   matlab -batch "addpath(genpath('<EasySpin>/easyspin')); run('tests/data/generate_pepper_quadzeeman_refs.m')"
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
Sys = struct('S',1,'g',[2.00 2.02 2.04],'D',[300 30],'lwpp',1);
Sys.Ham220 = 2e-4; Sys.Ham222 = [2e-4 0 3e-4 0 0];
Exp = struct('mwFreq',9.5,'Range',[150 500],'nPoints',4096);
cases{end+1} = mk('S1_ham2_lwpp',Sys,Exp,struct());
Sys2 = rmfield(Sys,'lwpp'); Sys2.HStrain = [40 50 60];
Exp.Harmonic = 0;
cases{end+1} = mk('S1_ham2_hstrain',Sys2,Exp,struct());
Sys3 = struct('S',3/2,'g',[2.0 2.0 2.0],'D',[1500 100],'lwpp',2);
Sys3.Ham220 = 1.5e-4; Sys3.Ham222 = [1e-4 0 -2e-4 0 0];
Exp3 = struct('mwFreq',9.5,'Range',[50 650],'nPoints',4096);
cases{end+1} = mk('S32_ham2_lwpp',Sys3,Exp3,struct());
% Frequency sweep. Needs EasySpin with the resfreqs_matrix.m Ham check (StollLab/EasySpin
% d46e5ef9, 2026-09-26): older copies test for 'ZB*' fields, miss Sys.Ham2xx and fail
% ("Wrong number of output arguments" from ham).
Sys4 = Sys; Sys4.lwpp = 30;
Exp4 = struct('Field',340,'mwRange',[8.5 10.5],'nPoints',4096,'Harmonic',0);
cases{end+1} = mk('S1_ham2_freqsweep',Sys4,Exp4,struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_quadzeeman.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
