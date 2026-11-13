% Stick spectra (no line width, Harmonic 0) whose SOPHE projection leaves
% empty bins (B47): EasySpin's projecttriangles.c / projectzones.c add every
% bin directly, so bins that no triangle or zone reaches are exactly 0.
% Reference for torchspin/tests/test_pepper_exactzero.py.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
Sys = struct('S',1/2,'g',[2 2.05 2.1]);
ExpB = struct('mwFreq',9.5,'Range',[325 337],'nPoints',601,'Harmonic',0);
% rhombic field sweep, no interpolation and 4x interpolation
cases{end+1} = mk('rhombic_field_31x1',Sys,ExpB,struct('GridSize',[31 1]));
cases{end+1} = mk('rhombic_field_31x4',Sys,ExpB,struct('GridSize',[31 4]));
% rhombic frequency sweep
ExpF = struct('Field',340,'mwRange',[9.4 10.2],'nPoints',601,'Harmonic',0);
cases{end+1} = mk('rhombic_freq_31x1',Sys,ExpF,struct('GridSize',[31 1]));
% axial (Dinfh, projectzones), field and frequency sweep
SysA = struct('S',1/2,'g',[2 2 2.1]);
cases{end+1} = mk('axial_field_31x1',SysA,struct('mwFreq',9.5,'Range',[315 345],'nPoints',601,'Harmonic',0),struct('GridSize',[31 1]));
cases{end+1} = mk('axial_freq_31x1',SysA,ExpF,struct('GridSize',[31 1]));
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_exactzero.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
