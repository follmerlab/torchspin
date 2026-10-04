% generate_ref_hamiltonians_ext.m
%
% Extended MATLAB/EasySpin reference set for torchspin Hamiltonian validation.
% Each case mirrors one EasySpin test in EasySpin/tests/ham_*.m and stores the
% spin system struct plus the matrices that test compares.  All random inputs
% are drawn here and stored inside Sys, so the Python side rebuilds the exact
% same system.
%
% Run from tests/data/ with EasySpin on the path:
%   matlab -batch "cd('tests/data'); addpath(genpath('<EasySpin>/easyspin')); generate_ref_hamiltonians_ext"
%
% Output: ref_ham_ext.mat (v7) with a cell array `cases`, each a struct with
% fields  name, Sys, [B0], and matrix fields (H0, mux, muy, muz, H, Hzf, Hhf,
% Hee, Hnq, Hnn, Hcf, Hso, Hoz, Hezho, ...).  Field `note` documents any
% deviation from the EasySpin test (e.g. an unsupported shorthand replaced by
% its expanded form).

clear cases
cases = {};
rng(20260901);

%% ---- ham_ee ------------------------------------------------------------

% ham_ee_dipolar: ee = d*[1 1 -2] + r*[1 -1 0]
Sys = struct('S',[1/2 1/2],'g',[2 2]);
d = 2.3e5; r = 1.87454e4;
Sys.ee = d*[1 1 -2] + r*[1 -1 0];
cases{end+1} = mkcase('ee_dipolar',Sys,[],struct('Hee',ham_ee(Sys)));

% ham_ee_full_single / ham_ee_full: three spins, full ee, pair selection
Sys = struct('S',[1/2 1/2 1/2]);
Sys.ee = [diag([1 1 1]); diag([2 2 2]); diag([5 5 5])];
ex = struct('Hee',ham_ee(Sys),'Hee_12',ham_ee(Sys,[1 2]), ...
            'Hee_13',ham_ee(Sys,[1 3]),'Hee_23',ham_ee(Sys,[2 3]));
cases{end+1} = mkcase('ee_full_three',Sys,[],ex);

% ham_ee_isotropic: one scalar per pair (column) vs principal rows
Sys = struct('S',[1/2 1/2 1/2]);
Sys.ee = [100 121 37];   % EasySpin: nPairs==3 → isotropic per pair
cases{end+1} = mkcase('ee_isotropic_perpair',Sys,[],struct('Hee',ham_ee(Sys)));
Sys.ee = [100 121 37].'*[1 1 1];
cases{end+1} = mkcase('ee_isotropic_rows',Sys,[],struct('Hee',ham_ee(Sys)));

% ham_ee_JdD: full ee = J*eye + diag(D) + antisymmetric(d)
Sys = struct('S',[1/2 1/2],'g',[2 2]);
J = rand*1e5; dv = rand(1,3)*1e5; D = [-1 -1 2]*1e5;
Sys.ee = J*eye(3) + diag(D) + [0 dv(3) -dv(2); -dv(3) 0 dv(1); dv(2) -dv(1) 0];
cases{end+1} = mkcase('ee_JdD_full',Sys,[],struct('Hee',ham_ee(Sys)));

% ham_ee_manyspins (without ee2 — biquadratic not in torchspin)
Sys = struct('S',[1/2 1/2 1/2]);
Sys.ee = rand(3,1);
c = mkcase('ee_manyspins_column',Sys,[],struct('Hee',ham_ee(Sys)));
c.note = 'EasySpin test also uses Sys.ee2 (biquadratic); dropped here.';
cases{end+1} = c;

% ham_ee_threespins (explicit reference matrices in EasySpin test)
Sys = struct('S',[1/2 1/2 1/2],'g',[2 2 2; 2.1 2.1 2.1; 2.2 2.2 2.2]);
Sys.ee = [1 1 1; 2 2 2; 3 3 3];
ex = struct('Hee',ham_ee(Sys),'Hee_12',ham_ee(Sys,[1 2]), ...
            'Hee_13',ham_ee(Sys,[1 3]),'Hee_23',ham_ee(Sys,[2 3]));
cases{end+1} = mkcase('ee_threespins',Sys,[],ex);

% ham_ee_tilt: tilted g and ee frames, field rotated accordingly
gv = [2 2.1 2.2]; Jt = 100; dip = [-2 1 1]*30;
frame2 = [-15 90 0]*pi/180; R = erot(frame2); B = 400*[0;0;1];
Sys = struct('S',[1/2 1/2],'g',[gv; gv]); Sys.ee = Jt + dip;
cases{end+1} = mkcase('ee_tilt_untilted',Sys,B.',struct());
SysB = Sys; SysB.gFrame = [frame2; frame2]; SysB.eeFrame = frame2;
cases{end+1} = mkcase('ee_tilt_tilted',SysB,(R.'*B).',struct());

% ham_ee_twospins
Sys = struct('S',[1/2 1/2],'g',[2 2 2; 2 2 2]); Sys.ee = [3 4 5];
cases{end+1} = mkcase('ee_twospins',Sys,[],struct('Hee',ham_ee(Sys)));

% ham_ee_twoSys: four spins, one scalar per pair (no ee2)
Sys = struct('S',[1/2 1/2 1/2 1/2]); Sys.ee = rand(6,1);
c = mkcase('ee_fourspins_column',Sys,[],struct('Hee',ham_ee(Sys)));
c.note = 'ee2 dropped (not supported by torchspin).';
cases{end+1} = c;

%% ---- ham_ez ------------------------------------------------------------

% ham_ez_fieldgiven: full (non-symmetric) g, random field
Sys = struct('S',3/2); Sys.g = diag([2 2 2]) + rand/2;
B0 = (ang2vec(rand*2*pi,rand*pi)*340).';
[mx,my,mz] = ham_ez(Sys);
cases{end+1} = mkcase('ez_fieldgiven_fullg',Sys,B0, ...
    struct('mux_e',mx,'muy_e',my,'muz_e',mz,'Hez',ham_ez(Sys,B0)));

% ham_ez_fullg_angles: full g with gFrame, S=3/2 + 1H
B0 = rand(1,3)*340;
Sys = struct('S',3/2,'Nucs','1H','A',[30 40 50]);
g = rand(3); gFrame = rand(1,3)*2*pi;
Sys.g = g; Sys.gFrame = gFrame;
cases{end+1} = mkcase('ez_fullg_angles',Sys,B0,struct('Hez',ham_ez(Sys,B0)));
Rg = erot(gFrame); Sys.g = Rg.'*g*Rg; Sys.gFrame = [0 0 0];
cases{end+1} = mkcase('ez_fullg_angles_rotated',Sys,B0,struct('Hez',ham_ez(Sys,B0)));

% ham_ez_simple
Sys = struct('S',1/2,'g',[2 2.1 2.2]);
[mx,my,mz] = ham_ez(Sys);
cases{end+1} = mkcase('ez_simple',Sys,[],struct('mux_e',mx,'muy_e',my,'muz_e',mz));

% ham_ez_twoelectrons: electron selection
Sys = struct('S',[1/2 1],'g',[2 2.1 2.2; 3 4 5],'ee',[1 2 3]);
[mx,my,mz] = ham_ez(Sys,2);
cases{end+1} = mkcase('ez_twoelectrons_select2',Sys,[], ...
    struct('mux_e2',mx,'muy_e2',my,'muz_e2',mz));

%% ---- ham_ezho / Stevens equivalence -------------------------------------

Alm = alm_table();

% ham_ezho_B0: Ham0kk vs Bk for S=9/2
Sys = struct('S',9/2); Sys2 = struct('S',9/2);
for lS = 2:2:8
  len = 2*lS+1;
  v = rand(1,len);
  Sys.(sprintf('Ham0%i%i',lS,lS)) = v;
  Sys2.(sprintf('B%i',lS)) = v./Alm(lS,1:len);
end
cases{end+1} = mkcase('ezho_B0_ham',Sys,[0 0 0],struct('Hezho',ham_ezho(Sys,[0 0 0])));
cases{end+1} = mkcase('ezho_B0_stevens',Sys2,[],struct('Hzf',ham_zf(Sys2)));

% ham_ezho_B1 / ham_full_zeemanVSzeemanho: Ham110/Ham112 vs full g
B0 = rand(1,3); S = 5/2;
Ham110 = rand; Ham112 = rand(5,1);
g = zeros(3);
g(1,2) = Ham112(5)/sqrt(2); g(1,3) = Ham112(2)/sqrt(2); g(2,3) = Ham112(4)/sqrt(2);
g = g + g';
g(3,3) = (-Ham110+sqrt(2)*Ham112(3))/sqrt(3);
g(2,2) = (-sqrt(2)*Ham110-Ham112(3)-sqrt(3)*Ham112(1))/sqrt(6);
g(1,1) = (-sqrt(2)*Ham110-Ham112(3)+sqrt(3)*Ham112(1))/sqrt(6);
g = g*(planck*1e9)/bmagn;
Sys = struct('S',S); Sys.Ham110 = Ham110; Sys.Ham112 = Ham112;
zHo = ham_ezho(Sys);
cases{end+1} = mkcase('ezho_B1_ham',Sys,B0, ...
    struct('Hezho',ham_ezho(Sys,B0),'G1x',zHo{2}{1},'G1y',zHo{2}{2},'G1z',zHo{2}{3}));
Sys2 = struct('S',S,'g',g);
[mx,my,mz] = ham_ez(Sys2);
cases{end+1} = mkcase('ezho_B1_fullg',Sys2,B0, ...
    struct('Hez',ham_ez(Sys2,B0),'mux_e',mx,'muy_e',my,'muz_e',mz));

% ham_ezho_tensors: random Ham fields for S=3/2
Sys = struct('S',3/2);
for lB = 0:3
  for lS = 1:2*Sys.S
    if mod(lB+lS,2)~=0, continue; end
    for l = abs(lB-lS):(lB+lS)
      if rand<0.4
        Sys.(sprintf('Ham%i%i%i',lB,lS,l)) = rand(1,2*l+1);
      end
    end
  end
end
B0 = rand(1,3);
[G0,G1,~,~] = ham_ezho(Sys);
cases{end+1} = mkcase('ezho_tensors',Sys,B0, ...
    struct('Hezho',ham_ezho(Sys,B0),'G0',G0,'G1x',G1{1},'G1y',G1{2},'G1z',G1{3}));

% ham_full_stevensVShigherorder: S=11/2
Sys = struct('S',11/2); Sys2 = struct('S',11/2);
for lS = 2:2:8
  len = 2*lS+1; v = rand(1,len);
  Sys.(sprintf('Ham0%i%i',lS,lS)) = v;
  Sys2.(sprintf('B%i',lS)) = v./Alm(lS,1:len);
end
cases{end+1} = mkcase('full_stevens_vs_ho_ham',Sys,[0 0 0],struct());
cases{end+1} = mkcase('full_stevens_vs_ho_stevens',Sys2,[0 0 0],struct());

%% ---- ham (full) ----------------------------------------------------------

% ham_full_syntax
Sys = struct('S',1/2,'g',[2 3 4]); B0 = rand(1,3)*400;
cases{end+1} = mkcase('full_syntax',Sys,B0,struct());

% ham_full_zeemanhfine: 63Cu with all terms
Sys = struct('S',1/2,'g',[2 3 4],'Nucs','63Cu','A',[50 50 350]); B0 = rand(1,3)*400;
[mxe,mye,mze] = ham_ez(Sys); [mxn,myn,mzn] = ham_nz(Sys);
cases{end+1} = mkcase('full_zeemanhfine_63Cu',Sys,B0, ...
    struct('Hnq',ham_nq(Sys),'Hhf',ham_hf(Sys),'mux_e',mxe,'muy_e',mye,'muz_e',mze, ...
           'mux_n',mxn,'muy_n',myn,'muz_n',mzn));

% ham_full_oam: two S=1/2 + L=1 centres with full g, soc, CF, Stevens, hyperfine
n = 2;
Sys = struct('S',1/2*ones(1,n));
Sys.g = rand(3*n,3);
Sys.L = ones(1,n);
Sys.soc = rand(n,2)*1000;
Sys.gL = rand(n,1);
Sys.ee = rand(1,1);
for k = 2:2:8
  Sys.(sprintf('B%d',k))  = rand(n,2*k+1).*repmat(((k/2)<=Sys.S).',1,2*k+1);
  Sys.(sprintf('CF%d',k)) = rand(n,2*k+1).*repmat(((k/2)<=Sys.L).',1,2*k+1);
end
Sys.Nucs = '1H,1H';
Sys.A = rand(3*2,3*n);
B0 = rand(1,3)*1e3;
c = mkcase('full_oam',Sys,B0,struct('Hcf',ham_cf(Sys),'Hso',ham_so(Sys),'Hzf',ham_zf(Sys),'Hhf',ham_hf(Sys)));
c.note = 'EasySpin test also has Sys.ee2 (biquadratic); dropped here.';
cases{end+1} = c;

%% ---- ham_hf ------------------------------------------------------------

% ham_hf_fullA_angles
A1 = rand(3); A2 = rand(3); AF1 = rand(1,3)*pi; AF2 = rand(1,3)*pi;
Sys = struct('S',1/2,'Nucs','1H,14N'); Sys.A = [A1; A2]; Sys.AFrame = [AF1; AF2];
cases{end+1} = mkcase('hf_fullA_angles',Sys,[],struct('Hhf',ham_hf(Sys)));
R1 = erot(AF1); R2 = erot(AF2);
Sys.A = [R1.'*A1*R1; R2.'*A2*R2]; Sys.AFrame = [0 0 0; 0 0 0];
cases{end+1} = mkcase('hf_fullA_angles_rotated',Sys,[],struct('Hhf',ham_hf(Sys)));

% ham_hf_fullA: four layouts
Sys = struct('S',1/2,'Nucs','1H,1H'); Sys.A = [rand(3); rand(3)];
cases{end+1} = mkcase('hf_fullA_1e2n',Sys,[],struct('Hhf',ham_hf(Sys)));
Sys = struct('S',[1/2 1/2],'Nucs','1H'); Sys.A = [rand(3) rand(3)]; Sys.ee = 1;
cases{end+1} = mkcase('hf_fullA_2e1n',Sys,[],struct('Hhf',ham_hf(Sys)));
Sys = struct('S',[1/2 1/2],'Nucs','1H,1H'); Sys.A = [rand(3) rand(3); rand(3) rand(3)]; Sys.ee = 1;
cases{end+1} = mkcase('hf_fullA_2e2n',Sys,[],struct('Hhf',ham_hf(Sys)));
Sys = struct('S',1/2,'Nucs','12C'); Sys.A = rand(3);
cases{end+1} = mkcase('hf_fullA_spin0nucleus',Sys,[],struct('Hhf',ham_hf(Sys)));

% ham_hf_isotropic
Sys = struct('S',1/2,'Nucs','1H,1H,1H','A',[100 121 37]);
cases{end+1} = mkcase('hf_isotropic_scalars',Sys,[],struct('Hhf',ham_hf(Sys)));
Sys.A = [100 121 37].'*[1 1 1];
cases{end+1} = mkcase('hf_isotropic_rows',Sys,[],struct('Hhf',ham_hf(Sys)));

% ham_hf_isotropic2: two electrons, three nuclei (expanded kron form)
aiso = [100 121; 34 56; 2 3];
Sys = struct('S',[1/2 1/2],'ee',1,'Nucs','1H,1H,1H'); Sys.A = kron(aiso,[1 1 1]);
c = mkcase('hf_isotropic2_2e3n',Sys,[],struct('Hhf',ham_hf(Sys)));
c.note = 'EasySpin also accepts A as (nNuclei x nElectrons) scalars; torchspin needs the kron-expanded form.';
cases{end+1} = c;

% ham_hf_selected: electron 2 / nucleus 2 selection
Sys = struct('S',[1/2 1 1/2],'Nucs','1H,1H'); Sys.A = rand(2,9); Sys.ee = [1 1 1];
cases{end+1} = mkcase('hf_selected',Sys,[],struct('Hhf',ham_hf(Sys),'Hhf_e2n2',ham_hf(Sys,2,2)));

% ham_hf_simplevalues
Sys = struct('S',1/2,'Nucs','1H','A',[10 12 -44],'g',2);
cases{end+1} = mkcase('hf_simplevalues',Sys,[],struct('Hhf',ham_hf(Sys)));

% ham_hf_tiltedvalues
Sys = struct('S',1/2,'Nucs','1H','g',[2 2 3]); Sys.A = rand(1,3); Sys.AFrame = rand(1,3)*2*pi;
cases{end+1} = mkcase('hf_tiltedvalues',Sys,[],struct('Hhf',ham_hf(Sys)));

%% ---- ham_nn ------------------------------------------------------------

% ham_nn_basic: scalar / principal / full
Sys = struct('S',1/2,'Nucs','1H,13C','A',[10 3]);
Sys.nn = 3;
cases{end+1} = mkcase('nn_basic_scalar',Sys,[],struct('Hnn',ham_nn(Sys)));
Sys.nn = [4 6 7];
cases{end+1} = mkcase('nn_basic_principal',Sys,[],struct('Hnn',ham_nn(Sys)));
Sys.nn = [1 2 3; 4 5 6; 7 8 9];
cases{end+1} = mkcase('nn_basic_full',Sys,[],struct('Hnn',ham_nn(Sys)));

% ham_nn_syntaxthree: three nuclei, three layouts, pair selection
Sys = struct('S',1/2,'Nucs','1H,13C,15N','A',[10 20 30]);
Sys.nn = [1 2 3];
cases{end+1} = mkcase('nn_three_perpair',Sys,[], ...
    struct('Hnn',ham_nn(Sys),'Hnn_12',ham_nn(Sys,[1 2]),'Hnn_13',ham_nn(Sys,[1 3]),'Hnn_23',ham_nn(Sys,[2 3])));
Sys.nn = [1 1 1; 2 2 2; 3 3 3];
cases{end+1} = mkcase('nn_three_rows',Sys,[], ...
    struct('Hnn',ham_nn(Sys),'Hnn_12',ham_nn(Sys,[1 2]),'Hnn_13',ham_nn(Sys,[1 3]),'Hnn_23',ham_nn(Sys,[2 3])));
Sys.nn = rand(9,3);
cases{end+1} = mkcase('nn_three_full',Sys,[], ...
    struct('Hnn',ham_nn(Sys),'Hnn_12',ham_nn(Sys,[1 2]),'Hnn_13',ham_nn(Sys,[1 3]),'Hnn_23',ham_nn(Sys,[2 3])));

%% ---- ham_nq ------------------------------------------------------------

% ham_nq_nonqi: spin-1/2 nucleus, no quadrupole
Sys = struct('S',1/2,'g',[2 2 2],'Nucs','1H','A',[1 2 3]);
cases{end+1} = mkcase('nq_nonqi',Sys,[],struct('Hnq',ham_nq(Sys)));

% ham_nq_nqisyntax: eeqQ / [eeqQ eta] shorthand — stored in expanded principal form
eeqQ = 17; eta = 0.1; I = nucspin('2H');
Sys = struct('S',1/2,'Nucs','2H','A',1);
Sys.Q = eeqQ;
Hq1 = ham_nq(Sys);
Sys.Q = eeqQ/(4*I*(2*I-1))*[-(1-0), -(1+0), 2];
c = mkcase('nq_eeqQ_expanded',Sys,[],struct('Hnq',Hq1));
c.note = 'EasySpin input was Sys.Q = eeqQ (scalar); stored expanded since torchspin lacks the shorthand.';
cases{end+1} = c;
Sys.Q = [eeqQ eta];
Hq2 = ham_nq(Sys);
Sys.Q = eeqQ/(4*I*(2*I-1))*[-(1-eta), -(1+eta), 2];
c = mkcase('nq_eeqQ_eta_expanded',Sys,[],struct('Hnq',Hq2));
c.note = 'EasySpin input was Sys.Q = [eeqQ eta]; stored expanded.';
cases{end+1} = c;

% ham_nq_onenucleus
Sys = struct('S',1/2,'g',2,'Nucs','2H','A',[1 2 3]); Sys.Q = [4 6 8];
cases{end+1} = mkcase('nq_onenucleus',Sys,[],struct('Hnq',ham_nq(Sys)));

% ham_nq_tiltedtensor
Sys = struct('S',1/2,'g',[2 2 2],'Nucs','2H','A',[1 2 3]);
Sys.Q = [-1.5 -0.5 2]; Sys.QFrame = [30 50 80]*pi/180;
cases{end+1} = mkcase('nq_tiltedtensor',Sys,[],struct('Hnq',ham_nq(Sys)));

% ham_nq_twonuclei: nucleus selection
Sys = struct('S',1/2,'g',[2 2 2],'Nucs','2H,2H','A',[1 2 3; 4 5 6]); Sys.Q = [-1 0 2; -1 0 2];
cases{end+1} = mkcase('nq_twonuclei',Sys,[],struct('Hnq',ham_nq(Sys),'Hnq_2',ham_nq(Sys,2)));

%% ---- ham_nz ------------------------------------------------------------

% ham_nz_electronnucleus
Sys = struct('S',1/2,'Nucs','14N','g',[2 2.1 2.2],'A',[1 2 3]);
[mx,my,mz] = ham_nz(Sys,1);
cases{end+1} = mkcase('nz_electronnucleus',Sys,[],struct('mux_n',mx,'muy_n',my,'muz_n',mz));

% ham_nz_shielding
Sys = struct('S',1/2,'Nucs','1H'); Sys.sigma = [1 2 3]; Sys.sigmaFrame = [pi/3 pi/5 pi/7];
Sys.gnscale = 0.9; Sys.A = 0;
[mx,my,mz] = ham_nz(Sys);
cases{end+1} = mkcase('nz_shielding',Sys,[],struct('mux_n',mx,'muy_n',my,'muz_n',mz));

% ham_nz_twonuclei
Sys = struct('S',1/2,'Nucs','1H,14N'); Sys.A = [12 23 34; 3 5 7]; Sys.Q = [0 0 0; -1 -1 2];
B0 = (ang2vec(pi/3,pi/6)*300).';
cases{end+1} = mkcase('nz_twonuclei',Sys,B0, ...
    struct('Hnz',ham_nz(Sys,B0),'Hnz_1',ham_nz(Sys,B0,1),'Hnz_2',ham_nz(Sys,B0,2)));

%% ---- ham_oz / ham_so / ham_cf ------------------------------------------

% ham_oz_explicit
Sys = struct('S',1/2,'L',1,'g',2,'gL',5/3,'soc',1000);
cases{end+1} = mkcase('oz_explicit',Sys,[0 0 1],struct('Hoz',ham_oz(Sys,[0;0;1]),'Hso',ham_so(Sys)));

% ham_oz_zeromomentum
Sys = struct('S',1/2,'L',0,'soc',1);
cases{end+1} = mkcase('oz_zeromomentum',Sys,[0 0 1000],struct('Hoz',ham_oz(Sys,[0 0 1000])));

% ham_so_simple
Sys = struct('S',1/2,'L',1,'soc',4545);
cases{end+1} = mkcase('so_simple',Sys,[],struct('Hso',ham_so(Sys)));

% ham_so_eeint: random multi-centre SL system
nSp = 2; S = randi(3,1,nSp)/2; L = randi(2,1,nSp);
Sys = struct('S',S,'L',L); Sys.soc = rand(nSp,2); Sys.gL = rand(nSp,1); Sys.ee = 0;
cases{end+1} = mkcase('so_eeint_two_centres',Sys,[],struct('Hso',ham_so(Sys)));

% ham_cf_zero_angmom: one centre has L=0
Sys = struct('S',[3/2 1/2],'L',[1 0]); Sys.soc = [1e4; 0]; Sys.ee = 1e3;
stev2 = [0 0 100 0 0]; Sys.CF2 = [stev2; stev2*0];
c = mkcase('cf_zero_angmom',Sys,[],struct('Hcf',ham_cf(Sys),'Hso',ham_so(Sys)));
c.note = 'EasySpin test used Sys.J = 1e3 (isotropic exchange); expressed as Sys.ee = 1e3.';
cases{end+1} = c;

% ham_cf_zerofield: random SL system, full g, gL
nC = 2; S = randi(3,1,nC)/2; L = randi(3,1,nC);
Sys = struct('S',S,'L',L); Sys.soc = zeros(nC,1); Sys.g = rand(3*nC,3); Sys.gL = rand(nC,1); Sys.ee = 0;
cases{end+1} = mkcase('cf_zerofield',Sys,[],struct('Hcf',ham_cf(Sys)));

% ham_cf_zfield: CF2..CF8 masked by L
Sys = struct('S',[1/2 1],'L',[2 1]); Sys.soc = zeros(2,1); Sys.ee = 0;
for k = 2:2:8
  Sys.(sprintf('CF%d',k)) = rand(2,2*k+1).*repmat(((k/2)<=Sys.L).',1,2*k+1);
end
cases{end+1} = mkcase('cf_zfield',Sys,[],struct('Hcf',ham_cf(Sys)));

%% ---- ham_zf ------------------------------------------------------------

% ham_zf_bframe: S=2, D/E vs B2, with frames
S = 2; D = 1000; E = D*0.2;
SysD = struct('S',S,'D',[D E]); SysB = struct('S',S); SysB.B2 = [E 0 D/3 0 0];
cases{end+1} = mkcase('zf_bframe_D_noframe',SysD,[],struct('Hzf',ham_zf(SysD)));
cases{end+1} = mkcase('zf_bframe_B2_noframe',SysB,[],struct('Hzf',ham_zf(SysB)));
ang = [0 pi/7.5 0]; SysD.DFrame = ang; SysB.B2Frame = ang;
cases{end+1} = mkcase('zf_bframe_D_beta',SysD,[],struct('Hzf',ham_zf(SysD)));
cases{end+1} = mkcase('zf_bframe_B2_beta',SysB,[],struct('Hzf',ham_zf(SysB)));
ang = [pi/3 pi/7.5 pi/5]; SysD.DFrame = ang; SysB.B2Frame = ang;
cases{end+1} = mkcase('zf_bframe_D_abg',SysD,[],struct('Hzf',ham_zf(SysD)));
cases{end+1} = mkcase('zf_bframe_B2_abg',SysB,[],struct('Hzf',ham_zf(SysB)));

% ham_zf_bkq_twoelectrons
Sys = struct('S',[7/2 7/2]); Sys.B4 = [rand(1,9); rand(1,9)]; Sys.ee = 1e-70;
cases{end+1} = mkcase('zf_bkq_twoelectrons',Sys,[],struct('Hzf',ham_zf(Sys)));

% ham_zf_bkq: single B43 / B40 entries
Sys = struct('S',7/2); Sys.B4 = [0 rand 0 0 0 0 0 0 0];
cases{end+1} = mkcase('zf_bkq_B43',Sys,[],struct('Hzf',ham_zf(Sys)));
Sys.B4 = [0 0 0 0 rand 0 0 0 0];
cases{end+1} = mkcase('zf_bkq_B40',Sys,[],struct('Hzf',ham_zf(Sys)));

% ham_zf_d_expansion_two: two electrons, D given as [D 0] rows and as principal rows
Sys = struct('S',[3/2 1],'ee',1); Dv = 100*rand(1,2);
Sys.D = [Dv(1) 0; Dv(2) 0];
cases{end+1} = mkcase('zf_d_expansion_two_DE',Sys,[],struct('Hzf',ham_zf(Sys)));
Sys.D = Dv(:)*[-1 -1 2]/3;
cases{end+1} = mkcase('zf_d_expansion_two_principal',Sys,[],struct('Hzf',ham_zf(Sys)));
Sys.D = Dv(:);
c = mkcase('zf_d_expansion_two_scalar',Sys,[],struct('Hzf',ham_zf(Sys)));
c.note = 'D given as one scalar per electron (EasySpin: D with E=0).';
cases{end+1} = c;

% ham_zf_d_expansion: S=3/2, scalar D / [D 0] / principal
Sys = struct('S',3/2); Dv = 100*rand;
Sys.D = Dv;
c = mkcase('zf_d_expansion_scalar',Sys,[],struct('Hzf',ham_zf(Sys)));
c.note = 'D given as scalar (EasySpin: D with E=0).';
cases{end+1} = c;
Sys.D = [Dv 0];
cases{end+1} = mkcase('zf_d_expansion_DE',Sys,[],struct('Hzf',ham_zf(Sys)));
Sys.D = [-1 -1 2]/3*Dv;
cases{end+1} = mkcase('zf_d_expansion_principal',Sys,[],struct('Hzf',ham_zf(Sys)));

% ham_zf_de: S=1, [D E] / principal / B2
Dv = 100*rand; Ev = Dv*rand;
Sys = struct('S',1,'g',2,'D',[Dv Ev]);
cases{end+1} = mkcase('zf_de_DE',Sys,[],struct('Hzf',ham_zf(Sys)));
Sys.D = [-1 -1 2]/3*Dv + [1 -1 0]*Ev;
cases{end+1} = mkcase('zf_de_principal',Sys,[],struct('Hzf',ham_zf(Sys)));
Sys = struct('S',1,'g',2); Sys.B2 = [Ev 0 Dv/3 0 0];
cases{end+1} = mkcase('zf_de_B2',Sys,[],struct('Hzf',ham_zf(Sys)));

% ham_zf_fullmatrix_angles
Sys = struct('S',3/2); Dm = rand(3); DFrame = rand(1,3)*2*pi;
Sys.D = Dm; Sys.DFrame = DFrame;
cases{end+1} = mkcase('zf_fullmatrix_angles',Sys,[],struct('Hzf',ham_zf(Sys)));
Rd = erot(DFrame); Sys.D = Rd.'*Dm*Rd; Sys.DFrame = [0 0 0];
cases{end+1} = mkcase('zf_fullmatrix_angles_rotated',Sys,[],struct('Hzf',ham_zf(Sys)));

% ham_zf_rhombic
Sys = struct('S',1,'D',[3 1]);
cases{end+1} = mkcase('zf_rhombic',Sys,[],struct('Hzf',ham_zf(Sys)));

% ham_zf_simple
Sys = struct('S',3/2,'D',[-1 -1 2]);
cases{end+1} = mkcase('zf_simple',Sys,[],struct('Hzf',ham_zf(Sys)));

%% ---- save --------------------------------------------------------------
fprintf('Generated %d cases.\n',numel(cases));
save(fullfile(fileparts(mfilename('fullpath')),'ref_ham_ext.mat'),'cases','-v7');
fprintf('Saved ref_ham_ext.mat\n');

%% ---- helpers -----------------------------------------------------------
function c = mkcase(name,Sys,B0,extra)
  c = struct();
  c.name = name;
  c.Sys = Sys;
  try
    [H0,mux,muy,muz] = ham(Sys);
    c.H0 = H0; c.mux = mux; c.muy = muy; c.muz = muz;
  catch err
    % EasySpin ham.m fails for Ham0kk-only systems with 4 outputs (cell + matrix)
    c.note_ham = sprintf('EasySpin ham(Sys) 4-output form failed: %s',err.message);
    H0 = ham(Sys,[0 0 0]);
  end
  if ~isempty(B0)
    c.B0 = B0(:).';
    c.H = ham(Sys,B0);
  end
  fn = fieldnames(extra);
  for k = 1:numel(fn)
    c.(fn{k}) = extra.(fn{k});
  end
  fprintf('  %s (dim %d)\n',name,size(H0,1));
end

function Alms = alm_table()
  Alm(8,:) = [24*sqrt(1430),2*sqrt(1430),4*sqrt(143/7),2*sqrt(78/7), ...
    4*sqrt(130/7), 2*sqrt(10/7), 4*sqrt(15), 2*sqrt(2), 8*sqrt(2)];
  Alm(7,1:8) = [4*sqrt(429), 8*sqrt(429/7),4*sqrt(286/7),8*sqrt(143/7),...
    4*sqrt(13/7), 8*sqrt(13/7),4*sqrt(2/7),8];
  Alm(6,1:7) = [4*sqrt(231),2*sqrt(11),4*sqrt(22/5),2*sqrt(22/5),...
    4*sqrt(11/3), 2*sqrt(2/3), 4*sqrt(2)];
  Alm(5,1:6) = [6*sqrt(14), 2*sqrt(42/5),sqrt(6/5),12/sqrt(5), 2*sqrt(2/5), 4];
  Alm(4,1:5) = [2*sqrt(70), sqrt(7), sqrt(14), 1, 2*sqrt(2)];
  Alm(3,1:4) = [sqrt(10), 2*sqrt(5/3), sqrt(2/3),2];
  Alm(2,1:3) = [sqrt(6), 1/sqrt(2), sqrt(2)];
  Alm(1,1:2) = [1,1];
  for n = 8:-1:1
    Alms(n,1:2*n+1) = [fliplr(Alm(n,2:n+1)), Alm(n,1:n+1)];
  end
end
