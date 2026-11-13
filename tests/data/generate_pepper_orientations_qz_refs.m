% EasySpin references for pepper / pepper_autograd with Exp.Orientations and
% field-quadratic Sys.Ham2xx terms (torchspin Options.QuadraticZeeman=True,
% B64, 0.9.16), plus Exp.Orientations with Opt.Transitions. EasySpin always
% includes the Ham2xx terms in its level model (resfields builds it from
% ham(Sys,B)), so it needs no option. As in generate_pepper_orientations_refs.m,
% the chi average over B1 directions is an exact crystal emulation:
% Exp.MolFrame = [0 0 0], one Exp.SampleFrame row [0 theta pi-phi] per
% orientation and Exp.SampleRotation = {'z',[0 pi/2]}; EasySpin weights every
% crystal orientation by 1/(nSamples*nRotations) = torchspin's default 1/N.
% Run from the repo root:
%   matlab -batch "addpath(genpath('<EasySpin>/easyspin')); run('tests/data/generate_pepper_orientations_qz_refs.m')"
clear all
cases = {};
function c = mk(name,Sys,Exp,Opt,ori)
  Opt.Verbosity = 0;
  Exp.MolFrame = [0 0 0];
  Exp.SampleFrame = [zeros(size(ori,1),1) ori(:,2) pi-ori(:,1)];
  Exp.SampleRotation = {'z',[0 pi/2]};
  [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'ori',ori,'x',x,'spc',spc);
end
% 1-2. S = 1, D = [300 30] MHz, Ham220 + Ham222 (the B64 system), lwpp 1 mT
Sys = struct('S',1,'g',[2.00 2.02 2.04],'D',[300 30],'lwpp',1);
Sys.Ham220 = 2e-4; Sys.Ham222 = [2e-4 0 3e-4 0 0];
Exp = struct('mwFreq',9.5,'Range',[150 500],'nPoints',2048,'Harmonic',1);
cases{end+1} = mk('S1_ham2_ori1',Sys,Exp,struct(),[0.7 1.1]);
cases{end+1} = mk('S1_ham2_ori5',Sys,Exp,struct(),[0.3 0.7; 1.9 2.2; 4.0 1.1; 5.5 0.4; 2.7 1.6]);
% 3. HStrain instead of lwpp (per-line strain widths), absorption
Sys2 = rmfield(Sys,'lwpp'); Sys2.HStrain = [40 50 60];
Exp2 = Exp; Exp2.Harmonic = 0;
cases{end+1} = mk('S1_ham2_hstrain_ori5',Sys2,Exp2,struct(),[0.3 0.7; 1.9 2.2; 4.0 1.1; 5.5 0.4; 2.7 1.6]);
% 4. Opt.Transitions on Exp.Orientations: only the 1-2 level pair, no Ham2xx
Sys3 = struct('S',1,'g',[2.00 2.02 2.04],'D',[300 30],'lwpp',1);
cases{end+1} = mk('S1_transitions12_ori5',Sys3,Exp,struct('Transitions',[1 2]),[0.3 0.7; 1.9 2.2; 4.0 1.1; 5.5 0.4; 2.7 1.6]);
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_orientations_qz.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
