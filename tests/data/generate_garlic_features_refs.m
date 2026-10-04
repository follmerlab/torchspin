% garlic features (PORT_SPEC_2 item 4.1): isotope mixtures, multi-component input,
% Opt.separate='components', automatic sweep ranges, equivalent nuclei.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = garlic(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
Exp = struct('mwFreq',9.668,'CenterSweep',[345 2],'Harmonic',0);
cases{end+1} = mk('isotopemix_B',struct('Nucs','B','A',10,'lw',[0 0.05]),Exp,struct());
cases{end+1} = mk('isotopemix_11B',struct('Nucs','11B','A',10,'lw',[0 0.05]),Exp,struct());
cases{end+1} = mk('isotopemix_Cl_n2',struct('Nucs','Cl','n',2,'A',12,'lw',[0 0.05]),struct('mwFreq',9.668,'CenterSweep',[345 4],'Harmonic',0),struct());
% multi-component (garlic_multicomponents) with a natural-abundance component
Sys1 = struct('g',2,'Nucs','1H','lw',[0,0.1],'A',20,'weight',0.567);
Sys2 = struct('g',2,'Nucs','14N','lw',[0,0.03],'A',34,'weight',1.567);
Sys3 = struct('g',2,'Nucs','C','lw',[0,0.03],'A',5);
Exp = struct('mwFreq',9.7,'Range',[344 349]);
[x,y] = garlic({Sys1,Sys2,Sys3},Exp,struct('Verbosity',0));
cases{end+1} = struct('name','multicomponents_sum','Sys',Sys1,'Sys2',Sys2,'Sys3',Sys3,'Exp',Exp,'Opt',struct(),'x',x,'spc',y);
[x,y] = garlic({Sys1,Sys2,Sys3},Exp,struct('Verbosity',0,'separate','components'));
cases{end+1} = struct('name','multicomponents_separate','Sys',Sys1,'Sys2',Sys2,'Sys3',Sys3,'Exp',Exp,'Opt',struct('separate','components'),'x',x,'spc',y);
% automatic ranges (garlic_fieldrange, garlic_freqsweep_autorange)
cases{end+1} = mk('autorange_field',struct('g',2,'Nucs','1H','lw',[0,0.01],'A',50),struct('mwFreq',9.7),struct());
cases{end+1} = mk('autorange_field_N_H2',struct('g',2.003,'Nucs','14N,1H','n',[1 2],'A',[20 5],'lw',0.1),struct('mwFreq',9.7),struct());
cases{end+1} = mk('autorange_freq',struct('g',2,'Nucs','1H','A',100,'lw',10),struct('Field',340),struct());
cases{end+1} = mk('autorange_field_gauss',struct('g',2.0023,'Nucs','1H','A',30,'lw',[0.3 0]),struct('mwFreq',9.4),struct());
% equivalent nuclei vs explicit list (garlic_equivnuclei), perturb2
Exp = struct('mwFreq',9.669,'CenterSweep',[345 5],'Harmonic',0,'nPoints',1e4); Opt = struct('Method','perturb2');
cases{end+1} = mk('equiv_n3',struct('n',3,'lw',[0 0.02],'Nucs','1H','A',10),Exp,Opt);
cases{end+1} = mk('equiv_explicit3',struct('lw',[0 0.02],'Nucs','1H,1H,1H','A',[10 10 10]),Exp,Opt);
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_garlic_features.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
