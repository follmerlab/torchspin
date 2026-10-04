% Per-transition, per-orientation strain widths from EasySpin resfields for the
% multi-electron strain cases (same systems as generate_strain_multielectron_refs.m).
% Stores, for a handful of single-crystal orientations, the resonance fields,
% level pairs and widths (mT) so torchspin's strainwidth can be compared directly.
clear all
fprintf('Generating strain-width references...\n');
cases = {};

sysdefs = {};
S = struct('S',[1/2 1/2]); S.g = [2.00 2.05 2.15; 1.98 2.02 2.10]; S.gFrame = [0 0 0; pi/6 pi/4 pi/3];
S.gStrain = [0.01 0.02 0.03; 0.02 0.01 0.015]; S.ee = 300; S.lw = 0;
sysdefs{end+1} = {'two_spinhalf_gstrain_both', S, [290 380]};
S = struct('S',[1/2 1/2]); S.g = [2.00 2.00 2.00; 2.00 2.10 2.20]; S.gStrain = [0 0 0; 0.01 0.02 0.03]; S.ee = 1000; S.lw = 0;
sysdefs{end+1} = {'two_spinhalf_gstrain_second', S, [280 400]};
S = struct('S',1,'g',2.0); S.D = [600 80]; S.DFrame = [pi/5 pi/3 pi/7]; S.DStrain = [60 15]; S.lw = 0;
sysdefs{end+1} = {'triplet_dstrain_tilted', S, [250 430]};
S = struct('S',[1 1],'g',[2.0 2.0]); S.D = [500 60; 700 -90]; S.DStrain = [50 10; 40 20]; S.DStrainCorr = [0.5 -0.3]; S.ee = 50; S.lw = 0;
sysdefs{end+1} = {'two_triplets_dstrain_corr', S, [200 480]};
S = struct('S',1,'g',2.0); S.D = [800 150]; S.DStrain = [80 30]; S.DStrainCorr = 0.8; S.lw = 0;
sysdefs{end+1} = {'triplet_dstrain_corr', S, [250 430]};

% Field directions in the molecular frame (unit vectors), via SampleFrame with chi = 0
dirs = [0 0 1; 1 0 0; 0 1 0; 1 1 1; 0.3 -0.5 0.8; -0.7 0.2 0.4];
dirs = dirs ./ vecnorm(dirs,2,2);
for k = 1:numel(sysdefs)
  name = sysdefs{k}{1}; Sys = sysdefs{k}{2}; rng = sysdefs{k}{3};
  Exp = struct('mwFreq',9.5,'Range',rng);
  Opt = struct('Verbosity',0,'Threshold',0);
  oris = {};
  for j = 1:size(dirs,1)
    [phi,theta] = vec2ang(dirs(j,:));
    Exp.SampleFrame = [0 -theta -phi];   % lab z along dirs(j,:) in the molecular frame
    [Pos,Int,Wid,Trans] = resfields(Sys,Exp,Opt);
    % Verify the convention: field direction in molecular frame from SampleFrame
    R = erot(Exp.SampleFrame);
    nB_M = R.'*[0;0;1];
    o = struct('dir',dirs(j,:),'nB_M',nB_M.','Pos',Pos,'Int',Int,'Wid',Wid,'Trans',Trans);
    oris{end+1} = o; %#ok
  end
  cases{end+1} = struct('name',name,'Sys',Sys,'Exp',Exp,'oris',{oris}); %#ok
end
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'strain_ref_widths.mat'),'cases','-v7');
fprintf('Done: %d cases\n',numel(cases));
