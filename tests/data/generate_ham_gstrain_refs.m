% g strain with the linear Zeeman term given as Sys.Ham110/Ham112 (B56,
% 0.9.13). EasySpin divides gStrain by Sys.g, which validatespinsys sets to
% zero for a Ham110/Ham112 system, and returns an all-zero spectrum. torchspin
% divides by the g-equivalent of the Ham terms instead, so the references here
% are the same systems with that g given as Sys.g (and gFrame). The Ham
% spectra (without and with gStrain) are stored to document EasySpin's
% behavior: without strain they equal the g spectra. Reference for
% torchspin/tests/test_ham_handling.py.
clear all; cases = {};
function c = mk(name,SysG,SysH,Exp)
  Opt.Verbosity = 0;
  [x,spc] = pepper(SysG,Exp,Opt);
  SysH0 = rmfield(SysH,'gStrain');
  [~,spcHam0] = pepper(SysH0,Exp,Opt);   % Ham, no strain: equals the g spectrum without strain
  [~,spcHam] = pepper(SysH,Exp,Opt);     % Ham + gStrain: all zero in EasySpin
  c = struct('name',name,'SysG',SysG,'SysH',SysH,'Exp',Exp,'x',x,'spc',spc, ...
             'spcHam0',spcHam0,'spcHam',spcHam);
end
Exp = struct('mwFreq',9.5,'Range',[330 350],'nPoints',512);
gs = [0.01 0.005 0.002];
% g = [2.01 2.005 1.998] along the molecular axes (HAM_LIN in test_saffron_ham.py)
SysG = struct('S',1/2,'g',[2.01 2.005 1.998],'lw',0.5,'gStrain',gs);
SysH = struct('S',1/2,'Ham110',-48.58946418805701, ...
  'Ham112',[0.04948419845999526 0 -0.10856491814573295 0 0],'lw',0.5,'gStrain',gs);
cases{end+1} = mk('diag',SysG,SysH,Exp);
% the same g with principal axes gFrame = [0.3 0.4 0.5]; gStrain in that frame
SysG = struct('S',1/2,'g',[2.01 2.005 1.998],'gFrame',[0.3 0.4 0.5],'lw',0.5,'gStrain',gs);
SysH = struct('S',1/2,'Ham110',-48.589464188057015, ...
  'Ham112',[-0.013029584283266012 -0.06880297648212826 -0.08035836887783973 ...
            -0.03825653265796862 0.03755499926915795], ...
  'gFrame',[0.3 0.4 0.5],'lw',0.5,'gStrain',gs);
cases{end+1} = mk('gframe',SysG,SysH,Exp);
% with a 1H hyperfine coupling
SysG = struct('S',1/2,'g',[2.01 2.005 1.998],'Nucs','1H','A',[3 5 12],'lw',0.5,'gStrain',gs);
SysH = struct('S',1/2,'Ham110',-48.58946418805701, ...
  'Ham112',[0.04948419845999526 0 -0.10856491814573295 0 0], ...
  'Nucs','1H','A',[3 5 12],'lw',0.5,'gStrain',gs);
cases{end+1} = mk('diag_1H',SysG,SysH,Exp);
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_ham_gstrain.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
for k=1:numel(cases), c=cases{k}; fprintf('%s: max|spc| %g, Ham0 rel diff %g, max|spcHam| %g\n',c.name,max(abs(c.spc)),max(abs(c.spcHam0-pepper(rmfield(c.SysG,'gStrain'),c.Exp,struct('Verbosity',0))))/max(abs(c.spcHam0)),max(abs(c.spcHam))); end
