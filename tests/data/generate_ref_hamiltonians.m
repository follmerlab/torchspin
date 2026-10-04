% generate_ref_hamiltonians.m
% Generate MATLAB reference data for Hamiltonian validation tests.
% This creates .mat files that torchspin tests will load and compare against.

% Add EasySpin to path
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

datadir = fileparts(mfilename('fullpath'));

fprintf('Generating Hamiltonian reference data in %s\n', datadir);

%% Test 1: Simple Zeeman (ham_ez) - isotropic g, S=1/2
fprintf('  Test 1: Zeeman isotropic g...\n');
Sys1.S = 1/2;
Sys1.g = [2.0, 2.0, 2.0];
B0_1 = [0, 0, 350.0];  % mT
[H1_0, mux1, muy1, muz1] = ham(Sys1);
H1_ez = ham_ez(Sys1, B0_1);
H1_full = ham(Sys1, B0_1);
save(fullfile(datadir, 'ref_ham_ez_iso.mat'), 'H1_0', 'mux1', 'muy1', 'muz1', 'H1_ez', 'H1_full', 'B0_1');

%% Test 2: Anisotropic g - S=1/2
fprintf('  Test 2: Zeeman anisotropic g...\n');
Sys2.S = 1/2;
Sys2.g = [2.0, 2.1, 2.2];
Sys2.gFrame = [0, 0, 0];  % radians
B0_2 = [0, 0, 340.0];
[H2_0, mux2, muy2, muz2] = ham(Sys2);
H2_ez = ham_ez(Sys2, B0_2);
H2_full = ham(Sys2, B0_2);
save(fullfile(datadir, 'ref_ham_ez_aniso.mat'), 'H2_0', 'mux2', 'muy2', 'muz2', 'H2_ez', 'H2_full', 'B0_2');

%% Test 3: Zero-field splitting - S=1, axial D
fprintf('  Test 3: ZFS axial D for S=1...\n');
Sys3.S = 1;
Sys3.g = [2.0, 2.0, 2.0];
Sys3.D = [-100, -100, 200];  % MHz (Dxx, Dyy, Dzz)
Sys3.DFrame = [0, 0, 0];
[H3_0, mux3, muy3, muz3] = ham(Sys3);
H3_zf = ham_zf(Sys3);
H3_full = ham(Sys3);
save(fullfile(datadir, 'ref_ham_zf_axial.mat'), 'H3_0', 'mux3', 'muy3', 'muz3', 'H3_zf', 'H3_full');

%% Test 4: ZFS rhombic - S=1
fprintf('  Test 4: ZFS rhombic D for S=1...\n');
Sys4.S = 1;
Sys4.g = [2.0, 2.0, 2.0];
Sys4.D = [-150, -50, 200];  % MHz
Sys4.DFrame = [0, 0, 0];
[H4_0, mux4, muy4, muz4] = ham(Sys4);
H4_zf = ham_zf(Sys4);
H4_full = ham(Sys4);
save(fullfile(datadir, 'ref_ham_zf_rhombic.mat'), 'H4_0', 'mux4', 'muy4', 'muz4', 'H4_zf', 'H4_full');

%% Test 5: Hyperfine - S=1/2, I=1 (14N)
fprintf('  Test 5: Hyperfine S=1/2, I=1...\n');
Sys5.S = 1/2;
Sys5.g = [2.006, 2.006, 2.002];
Sys5.Nucs = '14N';
Sys5.A = [20, 20, 90];  % MHz
Sys5.AFrame = [0, 0, 0];
[H5_0, mux5, muy5, muz5] = ham(Sys5);
H5_hf = ham_hf(Sys5);
H5_full = ham(Sys5);
save(fullfile(datadir, 'ref_ham_hf_14N.mat'), 'H5_0', 'mux5', 'muy5', 'muz5', 'H5_hf', 'H5_full');

%% Test 6: Hyperfine - S=1/2, two nuclei (14N + 1H)
fprintf('  Test 6: Hyperfine S=1/2, two nuclei...\n');
Sys6.S = 1/2;
Sys6.g = [2.006, 2.006, 2.002];
Sys6.Nucs = '14N,1H';
Sys6.A = [20, 20, 90; 5, 5, 15];  % MHz
Sys6.AFrame = [0, 0, 0; 0, 0, 0];
[H6_0, mux6, muy6, muz6] = ham(Sys6);
H6_hf = ham_hf(Sys6);
H6_full = ham(Sys6);
save(fullfile(datadir, 'ref_ham_hf_multiNuc.mat'), 'H6_0', 'mux6', 'muy6', 'muz6', 'H6_hf', 'H6_full');

%% Test 7: Electron-electron coupling - two S=1/2
fprintf('  Test 7: Electron-electron S=1/2 + S=1/2...\n');
Sys7.S = [1/2, 1/2];
Sys7.g = [2.0, 2.0, 2.0; 2.0, 2.0, 2.0];
Sys7.ee = [10, 10, 20];  % MHz
Sys7.eeFrame = [0, 0, 0];
[H7_0, mux7, muy7, muz7] = ham(Sys7);
H7_ee = ham_ee(Sys7);
H7_full = ham(Sys7);
save(fullfile(datadir, 'ref_ham_ee_twospin.mat'), 'H7_0', 'mux7', 'muy7', 'muz7', 'H7_ee', 'H7_full');

%% Test 8: Combined - S=1/2 with 14N, D, g-anisotropy, in field
fprintf('  Test 8: Combined terms...\n');
Sys8.S = 1/2;
Sys8.g = [2.009, 2.006, 2.002];
Sys8.gFrame = [0, 0, 0];
Sys8.Nucs = '14N';
Sys8.A = [15, 15, 80];
Sys8.AFrame = [0, 0, 0];
B0_8 = [0, 0, 335.0];
[H8_0, mux8, muy8, muz8] = ham(Sys8);
H8_full_noB = ham(Sys8);
H8_full_withB = ham(Sys8, B0_8);
H8_ez = ham_ez(Sys8, B0_8);
H8_hf = ham_hf(Sys8);
save(fullfile(datadir, 'ref_ham_combined.mat'), ...
     'H8_0', 'mux8', 'muy8', 'muz8', 'H8_full_noB', 'H8_full_withB', 'H8_ez', 'H8_hf', 'B0_8');

%% Test 9: High-spin S=5/2 with ZFS (e.g., Mn(II))
fprintf('  Test 9: High-spin S=5/2 with D...\n');
Sys9.S = 5/2;
Sys9.g = [2.0, 2.0, 2.0];
Sys9.D = [-200, -200, 400];  % MHz
[H9_0, mux9, muy9, muz9] = ham(Sys9);
H9_zf = ham_zf(Sys9);
H9_full = ham(Sys9);
save(fullfile(datadir, 'ref_ham_highspin.mat'), 'H9_0', 'mux9', 'muy9', 'muz9', 'H9_zf', 'H9_full');

%% Test 10: Frame rotations - tilted g tensor
fprintf('  Test 10: Tilted g-tensor frame...\n');
Sys10.S = 1/2;
Sys10.g = [2.0, 2.1, 2.2];
Sys10.gFrame = [pi/6, pi/4, pi/3];  % radians
B0_10 = [100, 200, 300];  % mT, general direction
[H10_0, mux10, muy10, muz10] = ham(Sys10);
H10_ez = ham_ez(Sys10, B0_10);
H10_full = ham(Sys10, B0_10);
save(fullfile(datadir, 'ref_ham_tilted.mat'), 'H10_0', 'mux10', 'muy10', 'muz10', 'H10_ez', 'H10_full', 'B0_10');

fprintf('Done! Generated 10 reference Hamiltonian files.\n');
