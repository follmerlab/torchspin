% Generate MATLAB references for the saffron phase/manifold/binning regressions.
%
% These cases were built to expose four defects that the pre-existing
% cosine-based saffron tests could not see (a DC-dominated cosine stays
% >0.99 while the modulation is ~100% wrong):
%
%   1. the conjugation of the second Q_ factor (tau phase) in the predefined
%      2p/3p/4pESEEM and HYSCORE dispatches,
%   2. the alpha/beta manifold assignment in the S=1/2 fast path (only
%      visible once a nuclear quadrupole makes the manifold eigenvectors
%      non-trivial),
%   3. the dimension order of the 2D frequency-bin accumulation,
%   4. the Opt.OriThreshold reference level under orientation selection
%      (S>1/2 with Exp.ExciteWidth).
%
% Every case therefore uses nonzero tau, anisotropic hyperfine couplings and
% (where the nucleus allows) a nuclear quadrupole, so that the modulation --
% not just the DC offset -- is compared. Run from the repository root with
% EasySpin on the MATLAB path:
%
%   matlab -nodisplay -batch "run('tests/data/generate_saffron_phase_refs.m')"

clear data

Sys14N = struct('S',0.5,'g',[2.0100 2.0050 1.9980],'Nucs','14N', ...
                'A',[2.5 3.1 5.4],'Q',[0.8 0.5]);
Sys13C = struct('S',0.5,'g',[2.00432 2.0036 2.00217],'Nucs','13C', ...
                'A',[-13.5 -9 40]);
SysS32 = struct('S',1.5,'g',[2.01 2.005 1.998],'D',50,'Nucs','14N', ...
                'A',[2.5 3.1 5.4],'Q',[-0.3 -0.1 0.4]);
G = struct('GridSize',13);

% --- 1D predefined sequences: direct time evolution and frequency binning
E2p = struct('Sequence','2pESEEM','Field',340,'dt',0.012,'nPoints',200,'tau',0.096);
E3p = struct('Sequence','3pESEEM','Field',340,'dt',0.012,'nPoints',200,'tau',0.096,'T',0.05);
E4p = struct('Sequence','4pESEEM','Field',340,'dt',0.012,'nPoints',200,'tau',0.096,'T',0.05);

Op = G; Op.TimeDomain = 1;
[~,data.y_2p_td] = saffron(Sys14N,E2p,Op);
[~,data.y_3p_td] = saffron(Sys14N,E3p,Op);
[~,data.y_4p_td] = saffron(Sys14N,E4p,Op);

Op = G; Op.TimeDomain = 0; Op.Expand = 4;
[~,data.y_2p_bin] = saffron(Sys14N,E2p,Op);
[~,data.y_3p_bin] = saffron(Sys14N,E3p,Op);
[~,data.y_4p_bin] = saffron(Sys14N,E4p,Op);

% --- HYSCORE: unequal t1/t2 and unequal nPoints so an axis swap cannot hide
Ehys = struct('Sequence','HYSCORE','Field',340,'dt',0.012, ...
              'nPoints',[64 64],'tau',0.096,'t1',0.02,'t2',0.03);
Op = G; Op.TimeDomain = 1;  [~,data.y_hys_td]  = saffron(Sys14N,Ehys,Op);
Op = G; Op.TimeDomain = 0; Op.Expand = 2;
[~,data.y_hys_bin] = saffron(Sys14N,Ehys,Op);

% Rectangular HYSCORE: nPoints(1) ~= nPoints(2) catches a transposed or
% mis-strided 2D binning buffer that a square grid would hide.
Erect = Ehys; Erect.nPoints = [32 64]; Erect.dt = [0.012 0.008];
Op = G; Op.TimeDomain = 1;  [~,data.y_hys_rect_td]  = saffron(Sys14N,Erect,Op);
Op = G; Op.TimeDomain = 0; Op.Expand = 2;
[~,data.y_hys_rect_bin] = saffron(Sys14N,Erect,Op);

% --- Orientation selection on (HStrain + ExciteWidth), 13C, no quadrupole
Eorisel = struct('mwFreq',34.7286,'Sequence','HYSCORE','tau',0.112, ...
                 'dt',0.008,'nPoints',64,'ExciteWidth',80,'Field',1238);
Sorisel = Sys13C; Sorisel.HStrain = [35 35 100];
Op = struct('GridSize',15,'TimeDomain',1,'Expand',2);
[~,data.y_orisel] = saffron(Sorisel,Eorisel,Op);

% --- S>1/2 with narrow excitation: exercises the OriThreshold reference level
Es32 = struct('Sequence','HYSCORE','Field',330,'dt',0.012,'nPoints',[48 48], ...
              'tau',0.096,'t1',0.02,'t2',0.03,'mwFreq',9.5,'ExciteWidth',100);
Op = struct('GridSize',11,'TimeDomain',1);
[~,data.y_s32_hys] = saffron(SysS32,Es32,Op);
Es32b = struct('Sequence','3pESEEM','Field',330,'dt',0.012,'nPoints',150, ...
               'tau',0.096,'T',0.05,'mwFreq',9.5,'ExciteWidth',100);
[~,data.y_s32_3p] = saffron(SysS32,Es32b,Op);

% --- Explicitly declared single crystal (P1) at a general orientation
Ep1 = Ehys; Ep1.SampleFrame = [0.3 0.7 0.2]; Ep1.CrystalSymmetry = 'P1';
Op = struct('TimeDomain',1);
[~,data.y_p1_tilt] = saffron(Sys14N,Ep1,Op);

% --- Custom (user-defined) pulse sequence, HYSCORE-like, with quadrupole.
% EasySpin divides a custom sequence's signal by max|pathwayprefactor|
% (saffron.m:1669); predefined sequences are not renormalized that way.
tauC = 0.080; pc90.Flip = pi/2; pc180.Flip = pi;
Ec = struct('Field',330);
Ec.Sequence = {pc90 tauC pc90 0 pc180 0 pc90 tauC};
Ec.nPoints = [32 32];
Ec.Dim1 = {'d2' 0.1};
Ec.Dim2 = {'d3' 0.1};
Oc = G; Oc.TimeDomain = 1;  [~,data.y_custom_td]  = saffron(Sys14N,Ec,Oc);
Oc = G; Oc.TimeDomain = 0; Oc.Expand = 2;
[~,data.y_custom_bin] = saffron(Sys14N,Ec,Oc);

% --- ProductRule x TimeDomain with two nuclei
Sys2N = struct('S',0.5,'g',[2.0100 2.0050 1.9980],'Nucs','14N,1H', ...
               'A',[2.5 3.1 5.4; 1.2 1.8 4.0],'Q',[0.8 0.5; 0 0]);
Eh2 = struct('Sequence','HYSCORE','Field',340,'dt',0.012,'nPoints',[48 48], ...
             'tau',0.096,'t1',0.02,'t2',0.03);
Op = G; Op.ProductRule = 1; Op.TimeDomain = 1;
[~,data.y_pr1_td1] = saffron(Sys2N,Eh2,Op);
Op = G; Op.ProductRule = 1; Op.TimeDomain = 0; Op.Expand = 2;
[~,data.y_pr1_td0] = saffron(Sys2N,Eh2,Op);
Op = G; Op.ProductRule = 0; Op.TimeDomain = 1;
[~,data.y_pr0_td1] = saffron(Sys2N,Eh2,Op);
Op = G; Op.ProductRule = 0; Op.TimeDomain = 0; Op.Expand = 2;
[~,data.y_pr0_td0] = saffron(Sys2N,Eh2,Op);

% --- Two components with a weight
Sc1 = struct('S',0.5,'g',[2.01 2.005 1.998],'Nucs','14N','A',[2.5 3.1 5.4],'Q',[0.8 0.5]);
Sc2 = struct('S',0.5,'g',[2.02 2.00 1.99],'Nucs','1H','A',[1.2 1.8 4.0],'weight',0.3);
Op = G; Op.TimeDomain = 1;
[~,data.y_twocomp] = saffron({Sc1,Sc2},E3p,Op);

% --- Mims ENDOR: single nucleus, so EasySpin's Opt.EndorMethod default is 0
SysEndor = struct('S',0.5,'g',[2.01 2.005 1.998],'Nucs','1H','A',[3 4 9], ...
                  'lwEndor',0.1);
Eendor = struct('Sequence','MimsENDOR','Field',340,'tau',0.1,'nPoints',200, ...
                'Range',[8 20]);
[~,data.y_endor_1nuc] = saffron(SysEndor,Eendor,G);

% Two nuclei, where EasySpin's default switches to EndorMethod 1
SysEndor2 = struct('S',0.5,'g',[2.01 2.005 1.998],'Nucs','1H,1H', ...
                   'A',[3 4 9; 1.5 2 4],'lwEndor',0.1);
[~,data.y_endor_2nuc] = saffron(SysEndor2,Eendor,G);

outFile = fullfile(fileparts(mfilename('fullpath')),'saffron_phase_refs.mat');
save(outFile,'data','-v7');
fprintf('wrote %s\n', outFile);
