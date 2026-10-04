% pepper features (PORT_SPEC_2 items 3.3, 3.7, 3.8 + dispersion): modamp, dispersion,
% separate components, natural-abundance isotopologues.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [B,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
end
% pepper_modamp
cases{end+1} = mk('modamp',struct('lwpp',0.5,'g',[2 2.02]),struct('mwFreq',9.7,'CenterSweep',[346 20],'ModAmp',5),struct());
cases{end+1} = mk('modamp_h2',struct('lwpp',0.5,'g',[2 2.02]),struct('mwFreq',9.7,'CenterSweep',[346 20],'ModAmp',2,'Harmonic',2),struct());
% dispersion (pepper_dispersion_fieldsweep): Lorentzian, mwPhase pi/2
cases{end+1} = mk('dispersion_L',struct('g',2,'lw',[0 0.5]),struct('mwFreq',9.5,'Range',[335 344],'Harmonic',0,'mwPhase',pi/2),struct());
cases{end+1} = mk('dispersion_GL_h1',struct('g',[2 2.1 2.2],'lw',[1 0.5]),struct('mwFreq',9.5,'Range',[290 350],'Harmonic',1,'mwPhase',0.6),struct());
% separate components (pepper_separatecomponents)
Sys1 = struct('g',[2 2.2],'lwpp',1,'weight',0.567); Sys2 = struct('g',[2.05 2.1 2.15],'lwpp',2,'weight',1.734); Exp = struct('mwFreq',9.7,'Range',[300 360]);
[B,spc] = pepper({Sys1,Sys2},Exp,struct('separate','components','Verbosity',0)); cases{end+1} = struct('name','separate_components','Sys',Sys1,'Sys2',Sys2,'Exp',Exp,'Opt',struct('separate','components'),'B',B,'spc',spc);
% isotopologues: pepper_iso_cu, iso_customabund, iso_twonucs, isotopes_two_Si, iso_12c
cases{end+1} = mk('iso_Cu',struct('S',1/2,'g',[2 2.2],'Nucs','Cu','A',[50 400],'lwpp',1),struct('mwFreq',9.5,'Range',[280 350]),struct());
Sys = struct('S',1/2,'g',[2 2.2],'Nucs','(63,65)Cu','A',[50 400],'Abund',[1 1],'lwpp',0.3);
cases{end+1} = mk('iso_customabund',Sys,struct('mwFreq',9.5,'Range',[280 350],'nPoints',1e4),struct('GridSize',61,'Method','perturb'));
cases{end+1} = mk('iso_twonucs',struct('S',1/2,'g',[2 2.2],'Nucs','Cu,N','A',[50 400; 30 30],'lwpp',0.3),struct('mwFreq',9.5,'Range',[280 350],'nPoints',1e4),struct('GridSize',91,'Method','perturb'));
cases{end+1} = mk('iso_two_Si',struct('Nucs','Si,Si','lwpp',0.1,'A',[5 5 5; 10 10 10]),struct('mwFreq',9.7,'CenterSweep',[346 10]),struct());
cases{end+1} = mk('iso_C',struct('Nucs','C','lwpp',0.1,'A',[30 30 60]),struct('mwFreq',9.7,'CenterSweep',[346 10]),struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_features.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
