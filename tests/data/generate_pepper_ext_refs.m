% Extended pepper MATLAB references (PORT_SPEC_2 Phase 1). Cases are appended
% per item; each case stores Sys/Exp/Opt and the spectrum. Saved to
% ref_pepper_ext.mat (-v7). Run from tests/data with EasySpin on the path.
clear all
fprintf('Generating extended pepper references...\n');
cases = {};
function_handle_note = 'all cases use pepper(Sys,Exp,Opt)';

% ---- 1.1 symmetry frames ----------------------------------------------------
Exp = struct('mwFreq',9.5,'Range',[300 360],'nPoints',1024,'Harmonic',1);
Opt = struct('GridSize',[19 4],'Verbosity',0);
Sys = struct('S',1/2,'g',[2.0 2.0 2.2],'gFrame',[0.3 0.7 0.2],'lw',1);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_tilted_axial_g','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Sys = struct('S',1/2,'g',[2.2 2.0 2.0],'lw',1);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_xaxial_g','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Sys = struct('S',1/2,'g',[2.0 2.2 2.0],'lw',1);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_yaxial_g','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Sys = struct('S',1/2,'g',[2.0 2.1 2.2],'gFrame',[0.3 0.7 0.2],'lw',1);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_tilted_rhombic_g','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Sys = struct('S',1/2,'g',[2.2 2.0 2.1],'lw',1);  % permuted rhombic
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_permuted_rhombic_g','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Sys = struct('S',1/2,'g',[2.0 2.0 2.2],'Nucs','1H','A',[30 30 80],'AFrame',[0 pi/2 0],'lw',1);  % axial g + perpendicular axial A -> D2h with frame
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_axial_g_perp_A','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Exp = struct('mwFreq',9.5,'Range',[250 430],'nPoints',1024,'Harmonic',1);
Sys = struct('S',1,'g',2.0,'D',800,'lw',1);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_triplet_axial_D','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Sys = struct('S',1,'g',2.0,'D',800,'DFrame',[0.4 0.9 0.1],'lw',1);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_triplet_axial_D_tilted','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Sys = struct('S',1,'g',2.0,'D',[800 120],'DFrame',[0.4 0.9 0.1],'lw',1);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','frame_triplet_rhombic_D_tilted','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
% pepper_axialinvariance system, Dinfh reference spectrum (perturb1, [19 5])
Sys = struct('g',[1.9 2.3],'lw',1);
Exp = struct('mwFreq',9.5,'Range',[285 365],'Harmonic',0);
Opt = struct('Method','perturb1','GridSize',[19 5],'GridSymmetry','Dinfh','Verbosity',0);
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','axialinvariance_Dinfh','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);
Opt.GridSymmetry = 'Ci';
[B,spc] = pepper(Sys,Exp,Opt); cases{end+1} = struct('name','axialinvariance_Ci','Sys',Sys,'Exp',Exp,'Opt',Opt,'B',B,'spc',spc);

script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_ext.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
