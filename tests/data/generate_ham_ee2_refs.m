% Sys.ee2 (biquadratic exchange) and Sys.D_ ([D E/D]) references: zero-field Hamiltonian + moment operators.
clear all; cases = {};
Sys = struct('S',[1 1],'g',[2 2.1],'ee',100,'ee2',35); [F,Gx,Gy,Gz] = ham(Sys);
cases{end+1} = struct('name','ee2_isotropic','Sys',Sys,'F',full(F),'Gx',full(Gx),'Gy',full(Gy),'Gz',full(Gz));
Sys = struct('S',[3/2 1],'g',[2 2],'ee',[10 20 30],'eeFrame',[0.3 0.7 1.1],'ee2',-12); [F,Gx,Gy,Gz] = ham(Sys);
cases{end+1} = struct('name','ee2_aniso','Sys',Sys,'F',full(F),'Gx',full(Gx),'Gy',full(Gy),'Gz',full(Gz));
Sys = struct('S',[1 1 1/2],'g',[2 2 2],'ee',[100 50 20],'ee2',[5 0 -3]); [F,Gx,Gy,Gz] = ham(Sys);
cases{end+1} = struct('name','ee2_three','Sys',Sys,'F',full(F),'Gx',full(Gx),'Gy',full(Gy),'Gz',full(Gz));
Sys = struct('S',1,'g',2,'D_',[900 0.2]); [F,Gx,Gy,Gz] = ham(Sys);
cases{end+1} = struct('name','D_single','Sys',Sys,'F',full(F),'Gx',full(Gx),'Gy',full(Gy),'Gz',full(Gz));
Sys = struct('S',[1 3/2],'g',[2 2],'D_',[900 0.2; -400 0.05],'ee',30); [F,Gx,Gy,Gz] = ham(Sys);
cases{end+1} = struct('name','D_two','Sys',Sys,'F',full(F),'Gx',full(Gx),'Gy',full(Gy),'Gz',full(Gz));
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_ham_ee2.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
