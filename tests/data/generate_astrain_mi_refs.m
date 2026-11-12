% AStrain effective nuclear mI (torchspin 0.9.25, bug B68). EasySpin resfields.m
% (766-788) labels every level once with the mI of nucleus 1: eigenvectors of
% kH0 - mean(Exp.Range)*kmuzM (field along the molecular z axis), levels sorted
% by energy, mI = <Iz>, a transition uses the mean over its two levels at every
% orientation, and the AStrain matrix enters as (G + gAStrainCorr*mI*AStrain).^2.
% For every case: pepper spectra (Harmonic 0 and 1, default options) and the
% resfields positions, widths, intensities and level pairs at ten field
% directions phi = 0.3, theta = 0:10:90 deg (one SampleFrame row [0 -theta -phi]
% each), all with the same Exp.Range. Reference for
% torchspin/tests/test_astrain_mi_model.py.
% Run: matlab -singleCompThread -batch "addpath(genpath('<easyspin>')); cd tests/data; generate_astrain_mi_refs"
clear all
fprintf('Generating AStrain mI references...\n');

cs = {};
N14 = struct('S',1/2,'g',[2 2 2.1],'Nucs','14N','A',[20 20 90],'lw',0.3,'AStrain',10);
cs{end+1} = {'N14', N14, [315 345]};
H1 = N14; H1.Nucs = '1H';
cs{end+1} = {'H1', H1, [315 345]};
Cu63 = struct('S',1/2,'g',[2.05 2.05 2.25],'Nucs','63Cu','A',[50 50 500],'lw',0.3, ...
              'AStrain',[10 10 50]);
cs{end+1} = {'Cu63', Cu63, [290 350]};
N14gS = N14; N14gS.gStrain = [0.005 0.005 0.01];
cs{end+1} = {'N14_gS', N14gS, [315 345]};
N14gSm = N14gS; N14gSm.gAStrainCorr = -1;
cs{end+1} = {'N14_gSm', N14gSm, [315 345]};
% tilted A frame (A strain matrix rotated to the molecular frame; mI still from molecular z)
Cu63t = Cu63; Cu63t.AFrame = [0 0.4 0];
cs{end+1} = {'Cu63_AFrame', Cu63t, [290 350]};

th = (0:10:90)*pi/180;
ori = [zeros(numel(th),1)+0.3, th(:)];   % [phi theta] rows

cases = struct([]);
Opt = struct('Verbosity',0);
for k = 1:numel(cs)
  name = cs{k}{1}; Sys = cs{k}{2}; rng = cs{k}{3};
  Exp = struct('mwFreq',9.5,'Range',rng,'nPoints',1024,'Harmonic',1);
  [B,spc1] = pepper(Sys,Exp,Opt);
  Exp.Harmonic = 0;
  [~,spc0] = pepper(Sys,Exp,Opt);
  ExpW = struct('mwFreq',9.5,'Range',rng, ...
                'SampleFrame',[zeros(size(ori,1),1), -ori(:,2), -ori(:,1)]);
  [Pos,Int,Wid,Tr] = resfields(Sys,ExpW,Opt);
  c = struct('name',name,'Sys',Sys,'Range',rng,'nPoints',1024,'B',B,'spc0',spc0,'spc1',spc1, ...
             'ori',ori,'Pos',Pos,'Int',Int,'Wid',Wid,'Tr',Tr);
  if isempty(cases), cases = c; else, cases(end+1) = orderfields(c,cases(1)); end %#ok<AGROW>
  fprintf('  %s: %d lines x %d orientations\n', name, size(Pos,1), size(Pos,2));
end
cases = cases(:);
save('ref_astrain_mi.mat','cases','-v7');
fprintf('Saved ref_astrain_mi.mat\n');
