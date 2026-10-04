% Parallel-mode (B1 || B0) references, PORT_SPEC_2 item 3.2.
clear all; cases = {};
Sys = struct('S',1,'g',2.0,'D',[900 120],'lw',2); Exp = struct('mwFreq',9.5,'Range',[50 450],'Harmonic',0,'nPoints',2048);
Exp.mwMode = 'parallel'; [B,spc] = pepper(Sys,Exp,struct('Verbosity',0)); cases{end+1} = struct('name','triplet_parallel','Sys',Sys,'Exp',Exp,'Opt',struct(),'B',B,'spc',spc);
Exp.mwMode = 'perpendicular'; [B,spc] = pepper(Sys,Exp,struct('Verbosity',0)); cases{end+1} = struct('name','triplet_perpendicular','Sys',Sys,'Exp',Exp,'Opt',struct(),'B',B,'spc',spc);
Sys = struct('S',5/2,'g',2.0,'D',[1500 300],'lw',3); Exp = struct('mwFreq',9.5,'Range',[20 500],'Harmonic',1,'nPoints',2048,'mwMode','parallel');
[B,spc] = pepper(Sys,Exp,struct('Verbosity',0)); cases{end+1} = struct('name','S52_parallel','Sys',Sys,'Exp',Exp,'Opt',struct(),'B',B,'spc',spc);
Sys = struct('S',1/2,'g',[2.0 2.1 2.3],'Nucs','63Cu','A',[50 50 500],'lw',1); Exp = struct('mwFreq',9.5,'Range',[250 380],'Harmonic',1,'mwMode','parallel');
[B,spc] = pepper(Sys,Exp,struct('Verbosity',0)); cases{end+1} = struct('name','Cu_parallel','Sys',Sys,'Exp',Exp,'Opt',struct(),'B',B,'spc',spc);
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_parallel.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
