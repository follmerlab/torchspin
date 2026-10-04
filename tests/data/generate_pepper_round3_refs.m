% PORT_SPEC_3 Phase 1: ordering in frequency sweeps, photoselection with perturbation
% theory, Exp.mwMode excitation modes, Opt.separate='transitions', automatic sweep ranges.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
% 1.1 ordering in a frequency sweep
cases{end+1} = mk('ordering_freq',struct('g',[2 2.05 2.2],'lw',10),struct('Field',340,'mwRange',[9 10.6],'Harmonic',0,'Ordering',3),struct());
cases{end+1} = mk('ordering_freq_neg',struct('g',[2 2.05 2.2],'lw',10),struct('Field',340,'mwRange',[9 10.6],'Harmonic',0,'Ordering',-2),struct());
% 1.2 photoselection with perturbation theory (S=1/2)
Sys = struct('g',[2 2.1 2.2],'lwpp',1,'tdm','z'); Exp = struct('mwFreq',9.5,'Range',[290 360],'Harmonic',0,'lightScatter',0.2);
for lb = {'parallel','perpendicular','unpolarized'}
  Exp.lightBeam = lb{1};
  cases{end+1} = mk(sprintf('photosel_pt_%s',lb{1}),Sys,Exp,struct('Method','perturb'));
  cases{end+1} = mk(sprintf('photosel_mx_%s',lb{1}),Sys,Exp,struct('Method','matrix'));
end
% 1.4 excitation modes (pepper_unpolarized with a fixed angle)
Sys = struct('g',[3 2],'lwpp',10); Exp = struct('mwFreq',9.5,'Range',[180 380]);
Exp.mwMode = {0.7 'unpolarized'}; c = mk('mwmode_unpol_mx',Sys,Exp,struct('Method','matrix')); c.Exp.mwMode_k = 0.7; c.Exp.mwMode_m = 'unpolarized'; c.Exp = rmfield(c.Exp,'mwMode'); cases{end+1} = c;
c = mk('mwmode_unpol_pt',Sys,Exp,struct('Method','perturb')); c.Exp.mwMode_k = 0.7; c.Exp.mwMode_m = 'unpolarized'; c.Exp = rmfield(c.Exp,'mwMode'); cases{end+1} = c;
Sys = struct('g',[2 2.1 2.3],'lwpp',2); Exp = struct('mwFreq',9.5,'Range',[280 360]);
for mode = {{0.7 'unpolarized'},{0.7 'circular+'},{0.7 'circular-'},{[0.3 0.8] 0.4},{'z' 'circular+'}}
  Exp.mwMode = mode{1}; k = mode{1}{1}; m2 = mode{1}{2};
  if ischar(m2), mstr = m2; else, mstr = 'linear'; end
  if ischar(k), kstr = k; else, kstr = strjoin(arrayfun(@(v) sprintf('%g',v),k,'UniformOutput',false),'_'); end
  for meth = {'matrix','perturb'}
    c = mk(sprintf('mwmode_%s_k%s_%s',mstr,kstr,meth{1}),Sys,Exp,struct('Method',meth{1}));
    c.Exp = rmfield(c.Exp,'mwMode'); c.Exp.mwMode_k = k; c.Exp.mwMode_m = m2; cases{end+1} = c;
  end
end
% crystal with unpolarized / circular excitation
Sys = struct('g',[2 2.1 2.3],'lwpp',1); Exp = struct('mwFreq',9.5,'Range',[280 360],'CrystalSymmetry','P1','MolFrame',[0.3 0.7 0.1]);
for mode = {{0.7 'unpolarized'},{0.7 'circular+'},{[0.3 0.8] 0.4}}
  Exp.mwMode = mode{1}; k = mode{1}{1}; m2 = mode{1}{2};
  if ischar(m2), mstr = m2; else, mstr = 'linear'; end
  c = mk(sprintf('mwmode_crystal_%s',mstr),Sys,Exp,struct()); c.Exp = rmfield(c.Exp,'mwMode'); c.Exp.mwMode_k = k; c.Exp.mwMode_m = m2; cases{end+1} = c;
end
% 1.3 separate transitions (pepper_separate_transitions)
Exp = struct('mwFreq',9.5,'Range',[330 360]); Opt = struct('separate','transitions','Method','perturb2');
cases{end+1} = mk('septrans_35Cl_pt2',struct('g',2,'Nucs','35Cl','A',10,'lwpp',0.2),Exp,Opt);
cases{end+1} = mk('septrans_ClCl_pt2',struct('g',2,'Nucs','Cl,Cl','A',[20 7],'lwpp',0.2),Exp,Opt);
Sys1 = struct('g',2,'Nucs','35Cl','A',10,'lwpp',0.2); Sys3 = struct('g',2.02,'lwpp',0.4);
[x,spc] = pepper({Sys1,Sys3},Exp,struct('separate','transitions','Method','perturb2','Verbosity',0));
cases{end+1} = struct('name','septrans_two_components','Sys',Sys1,'Sys2',Sys3,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
cases{end+1} = mk('septrans_triplet_matrix',struct('S',1,'g',2,'D',[300 10],'lw',0.5),struct('mwFreq',9.5,'Range',[300 380],'Harmonic',0),struct('separate','transitions','Method','matrix'));
cases{end+1} = mk('septrans_CuN_matrix',struct('S',1/2,'g',[2 2.2],'Nucs','63Cu,14N','A',[50 400; 30 30],'lwpp',0.5),struct('mwFreq',9.5,'Range',[280 350]),struct('separate','transitions','Method','matrix'));
% 1.5 automatic sweep range (pepper_fieldrange, pepper_freqsweep_autorange)
cases{end+1} = mk('autorange_field',struct('g',[2 2.1 2.2],'lw',1),struct('mwFreq',35),struct());
cases{end+1} = mk('autorange_field_Cu',struct('S',1/2,'g',[2 2.2],'Nucs','63Cu','A',[50 400],'lwpp',1),struct('mwFreq',9.5),struct());
cases{end+1} = mk('autorange_freq',struct('g',[2 2.05 2.01],'lwpp',10),struct('Field',340),struct());
cases{end+1} = mk('autorange_freq_strain',struct('g',[2 2.05 2.01],'gStrain',[0.01 0.005 0.002]),struct('Field',340),struct());
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_round3.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
