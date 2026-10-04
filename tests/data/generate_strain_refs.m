% Generate reference data for strain validation tests
% Run this in MATLAB to create reference .mat files for torchspin validation

clear

% =========================================================================
% Test 1: HStrain only (S=1/2, anisotropic g, residual broadening)
% =========================================================================
fprintf('Generating Test 1: HStrain only...\n');

Sys1.S = 1/2;
Sys1.g = [2.00, 2.05, 2.15];
Sys1.HStrain = [50, 100, 150];  % MHz
Sys1.lw = 0;  % No additional broadening

Exp1.mwFreq = 9.5;  % GHz
Exp1.Range = [320, 380];  % mT
Exp1.nPoints = 1024;
Exp1.Harmonic = 1;  % 1st derivative

Opt1.GridSize = 31;
Opt1.Verbosity = 0;

[B1, spc1] = pepper(Sys1, Exp1, Opt1);

save('strain_ref_hstrain.mat', 'Sys1', 'Exp1', 'Opt1', 'B1', 'spc1', '-v7');
fprintf('  Saved strain_ref_hstrain.mat\n');

% =========================================================================
% Test 2: gStrain only (S=1/2, g distribution)
% =========================================================================
fprintf('Generating Test 2: gStrain only...\n');

Sys2.S = 1/2;
Sys2.g = [2.00, 2.05, 2.15];
Sys2.gStrain = [0.01, 0.02, 0.03];  % Relative to g
Sys2.lw = 0;

Exp2.mwFreq = 9.5;
Exp2.Range = [320, 380];
Exp2.nPoints = 1024;
Exp2.Harmonic = 1;

Opt2.GridSize = 31;
Opt2.Verbosity = 0;

[B2, spc2] = pepper(Sys2, Exp2, Opt2);

save('strain_ref_gstrain.mat', 'Sys2', 'Exp2', 'Opt2', 'B2', 'spc2', '-v7');
fprintf('  Saved strain_ref_gstrain.mat\n');

% =========================================================================
% Test 3: DStrain (S=1 triplet state with ZFS)
% =========================================================================
fprintf('Generating Test 3: DStrain (S=1 system)...\n');

Sys3.S = 1;
Sys3.g = 2.0;
Sys3.D = [500, 100];  % MHz, [D E] notation
Sys3.DStrain = [50, 10];  % MHz, strain in D and E
Sys3.lw = 0;

Exp3.mwFreq = 9.5;
Exp3.Range = [280, 400];
Exp3.nPoints = 1024;
Exp3.Harmonic = 1;

Opt3.GridSize = 31;
Opt3.Verbosity = 0;

[B3, spc3] = pepper(Sys3, Exp3, Opt3);

save('strain_ref_dstrain.mat', 'Sys3', 'Exp3', 'Opt3', 'B3', 'spc3', '-v7');
fprintf('  Saved strain_ref_dstrain.mat\n');

% =========================================================================
% Test 4: gStrain + AStrain with correlation (S=1/2 + 14N)
% =========================================================================
fprintf('Generating Test 4: gStrain + AStrain with correlation...\n');

Sys4.S = 1/2;
Sys4.g = [2.006, 2.006, 2.002];
Sys4.Nucs = '14N';
Sys4.A = [20, 40, 80];  % MHz
Sys4.gStrain = [0.01, 0.01, 0.02];
Sys4.AStrain = [5, 10, 15];  % MHz
Sys4.gAStrainCorr = -1;  % Anticorrelated
Sys4.lw = 0;

Exp4.mwFreq = 9.5;
Exp4.Range = [320, 360];
Exp4.nPoints = 1024;
Exp4.Harmonic = 1;

Opt4.GridSize = 31;
Opt4.Verbosity = 0;

[B4, spc4] = pepper(Sys4, Exp4, Opt4);

save('strain_ref_gstrain_astrain.mat', 'Sys4', 'Exp4', 'Opt4', 'B4', 'spc4', '-v7');
fprintf('  Saved strain_ref_gstrain_astrain.mat\n');

% =========================================================================
% Test 5: Combined HStrain + gStrain (common real-world case)
% =========================================================================
fprintf('Generating Test 5: HStrain + gStrain combined...\n');

Sys5.S = 1/2;
Sys5.g = [2.00, 2.05, 2.15];
Sys5.HStrain = [30, 50, 70];  % MHz
Sys5.gStrain = [0.005, 0.01, 0.015];
Sys5.lw = 0;

Exp5.mwFreq = 9.5;
Exp5.Range = [320, 380];
Exp5.nPoints = 1024;
Exp5.Harmonic = 1;

Opt5.GridSize = 31;
Opt5.Verbosity = 0;

[B5, spc5] = pepper(Sys5, Exp5, Opt5);

save('strain_ref_combined.mat', 'Sys5', 'Exp5', 'Opt5', 'B5', 'spc5', '-v7');
fprintf('  Saved strain_ref_combined.mat\n');

% =========================================================================
% Test 6: Tilted g-frame with gStrain
% =========================================================================
fprintf('Generating Test 6: Tilted g-frame with gStrain...\n');

Sys6.S = 1/2;
Sys6.g = [2.00, 2.05, 2.15];
Sys6.gFrame = [pi/6, pi/4, pi/3];  % Euler angles
Sys6.gStrain = [0.01, 0.02, 0.03];
Sys6.lw = 0;

Exp6.mwFreq = 9.5;
Exp6.Range = [320, 380];
Exp6.nPoints = 1024;
Exp6.Harmonic = 1;

Opt6.GridSize = 31;
Opt6.Verbosity = 0;

[B6, spc6] = pepper(Sys6, Exp6, Opt6);

save('strain_ref_tilted_frame.mat', 'Sys6', 'Exp6', 'Opt6', 'B6', 'spc6', '-v7');
fprintf('  Saved strain_ref_tilted_frame.mat\n');

fprintf('\n✓ All strain reference files generated successfully!\n');
fprintf('  Run Python tests with: pytest torchspin/tests/test_strain_validation.py\n');
