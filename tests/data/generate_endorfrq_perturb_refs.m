% EasySpin references for endorfrq_perturb (B20 intensities and element
% order, B27 full 3x3 Q; 0.8.8). Reference for
% torchspin/tests/test_endorfrq_perturb_matlab_validation.py.
%
%   >> cd tests/data
%   >> generate_endorfrq_perturb_refs
%
% Each case is called once with all four orientations: Exp.SampleFrame has
% one row [0 -theta -phi] per orientation (MolFrame = [0 0 0]), so the field
% direction in the molecular frame is (sin t cos p, sin t sin p, cos t), the
% (phi, theta) convention of torchspin.endorfrq_perturb. This EasySpin version
% has no Exp.CrystalOrientation (p_crystalorientations.m ignores it).
% Saved per case: name, Sys, ori (nOri x 2, [phi theta] in rad),
%   f, i   (nTrans x nOri) EndorFreqs (MHz) and EndorIntensities, PerturbOrder 2
%   f1, i1 the same with Opt.PerturbOrder = 1
clear all; cases = {};
ori = [0.3 0.7; 0 0; 1.1 1.4; 2.0 0.3];   % [phi theta], rad
Field = 340;                               % mT
function c = mk(name,Sys,ori,Field)
  Exp = struct('Field',Field,'SampleFrame',[zeros(size(ori,1),1) -ori(:,2) -ori(:,1)]);
  [f,i] = endorfrq_perturb(Sys,Exp,struct('PerturbOrder',2));
  [f1,i1] = endorfrq_perturb(Sys,Exp,struct('PerturbOrder',1));
  c = struct('name',name,'Sys',Sys,'ori',ori,'Field',Field,'f',f,'i',i,'f1',f1,'i1',i1);
end
g = [2.0 2.05 2.1];
cases{end+1} = mk('H1_AFrame',struct('S',1/2,'g',g,'Nucs','1H','A',[5 8 12],'AFrame',[0.2 0.4 0.1]),ori,Field);
cases{end+1} = mk('N14_Q_QFrame',struct('S',1/2,'g',g,'Nucs','14N','A',[20 25 30], ...
  'Q',[-0.4 -0.6 1.0],'QFrame',[0.1 0.5 0.3]),ori,Field);
cases{end+1} = mk('H1_N14',struct('S',1/2,'g',g,'Nucs','1H,14N','A',[5 8 12; 20 25 30], ...
  'Q',[0 0 0; -0.4 -0.6 1.0]),ori,Field);
cases{end+1} = mk('Mn55_Q',struct('S',1/2,'g',g,'Nucs','55Mn','A',[250 260 280],'Q',[-1 -2 3]),ori,Field);
cases{end+1} = mk('H1_strong',struct('S',1/2,'g',[2 2 2],'Nucs','1H','A',[200 200 200]),ori,Field);
cases{end+1} = mk('N14_fullQ',struct('S',1/2,'g',g,'Nucs','14N','A',[20 25 30], ...
  'Q',[-0.4 0.1 0; 0.1 -0.6 0.2; 0 0.2 1.0]),ori,Field);
% full 3x3 A (non-symmetric) and full Q on the second of two nuclei
cases{end+1} = mk('H1_N14_fullA_fullQ',struct('S',1/2,'g',g,'Nucs','1H,14N', ...
  'A',[5 1 0.5; 1.5 8 -1; 0.2 -0.8 12; 20 2 1; 2 25 -3; 1 -3 30], ...
  'Q',[0 0 0; 0 0 0; 0 0 0; -0.4 0.1 0; 0.1 -0.6 0.2; 0 0.2 1.0]),ori,Field);
% S = 3/2 with D: four mS manifolds, checks the mI-major / mS-ascending order
cases{end+1} = mk('S32_D_N14',struct('S',3/2,'g',g,'D',[300 50],'Nucs','14N','A',[20 25 30], ...
  'Q',[-0.4 -0.6 1.0]),ori,Field);
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'endorfrq_perturb_refs.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
