% Strain and grid symmetry (B82): EasySpin's hamsymm has no strain term, so an
% axial g with a rhombic HStrain or gStrain gets the Dinfh grid (one meridian).
% Saves EasySpin's automatic group and pepper spectra with the automatic grid
% and with a forced D2h grid ([31 4] and [91 4]). Reference for
% torchspin/tests/test_grid_symmetry_strain.py.
% Run: matlab -singleCompThread -batch "addpath(genpath('<EasySpin>/easyspin')); cd tests/data; generate_hamsymm_strain_refs"
clear all; cases = {};
function c = mk(name,Sys,Exp)
  Opt = struct('GridSize',[31 4],'Verbosity',0);
  [x,spc_auto] = pepper(Sys,Exp,Opt);
  Opt.GridSymmetry = 'D2h'; [~,spc_d2h] = pepper(Sys,Exp,Opt);
  Opt.GridSize = [91 4]; [~,spc_d2h_91] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'group',hamsymm(Sys),'x',x, ...
             'spc_auto',spc_auto,'spc_d2h',spc_d2h,'spc_d2h_91',spc_d2h_91);
end
Exp = struct('mwFreq',9.5,'Range',[300 345],'nPoints',1024,'Harmonic',1);
cases{end+1} = mk('hstrain',struct('S',1/2,'g',[2.05 2.05 2.15],'lw',0.3,'HStrain',[200 20 20]),Exp);
cases{end+1} = mk('gstrain',struct('S',1/2,'g',[2.05 2.05 2.15],'lw',0.3,'gStrain',[0.02 0.002 0.002]),Exp);
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_hamsymm_strain.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
