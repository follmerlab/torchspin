% Generate MATLAB reference data for garlic validation
%
% This script generates reference spectra using MATLAB EasySpin garlic
% for comparison with the torchspin Python implementation.
%
% Run this script from the EasySpin root directory:
%   cd /path/to/EasySpin
%   matlab -nodisplay -nosplash -r "run('tests/data/generate_garlic_refs.m'); exit"

clear all;

% Ensure EasySpin is on the path
if ~exist('garlic', 'file')
    addpath(genpath('easyspin'));
end

fprintf('Generating garlic reference data...\n');

%% Test Case 1: No nuclei (single line)
fprintf('  Case 1: No nuclei...\n');

Sys1.S = 1/2;
Sys1.g = 2.006;
Sys1.lw = [0.5, 0.0];  % Gaussian only

Exp1.mwFreq = 9.5;  % GHz
Exp1.Range = [330, 350];  % mT
Exp1.nPoints = 512;
Exp1.Harmonic = 0;  % Absorption

[B1, spec1] = garlic(Sys1, Exp1);

% Save
ref1.B = B1;
ref1.spec = spec1;
ref1.Sys = Sys1;
ref1.Exp = Exp1;
ref1.description = 'No nuclei, isotropic g, absorption';

%% Test Case 2: Single nucleus 14N (triplet)
fprintf('  Case 2: 14N triplet...\n');

Sys2.S = 1/2;
Sys2.g = 2.006;
Sys2.Nucs = '14N';
Sys2.A = 16.0;  % MHz, isotropic
Sys2.lw = [0.3, 0.0];

Exp2.mwFreq = 9.5;
Exp2.Range = [333, 345];
Exp2.nPoints = 512;
Exp2.Harmonic = 1;  % First derivative

[B2, spec2] = garlic(Sys2, Exp2);

ref2.B = B2;
ref2.spec = spec2;
ref2.Sys = Sys2;
ref2.Exp = Exp2;
ref2.description = '14N triplet, first derivative';

%% Test Case 3: Anisotropic g, averaged
fprintf('  Case 3: Anisotropic g...\n');

Sys3.S = 1/2;
Sys3.g = [2.002, 2.006, 2.010];  % Anisotropic
Sys3.lw = [0.5, 0.0];

Exp3.mwFreq = 9.5;
Exp3.Range = [330, 350];
Exp3.nPoints = 256;
Exp3.Harmonic = 0;

[B3, spec3] = garlic(Sys3, Exp3);

ref3.B = B3;
ref3.spec = spec3;
ref3.Sys = Sys3;
ref3.Exp = Exp3;
ref3.description = 'Anisotropic g (averaged to isotropic)';

%% Test Case 4: 1H doublet
fprintf('  Case 4: 1H doublet...\n');

Sys4.S = 1/2;
Sys4.g = 2.003;
Sys4.Nucs = '1H';
Sys4.A = 10.0;  % MHz
Sys4.lw = [0.2, 0.0];

Exp4.mwFreq = 9.5;
Exp4.Range = [336, 342];
Exp4.nPoints = 512;
Exp4.Harmonic = 1;

[B4, spec4] = garlic(Sys4, Exp4);

ref4.B = B4;
ref4.spec = spec4;
ref4.Sys = Sys4;
ref4.Exp = Exp4;
ref4.description = '1H doublet';

%% Test Case 5: Multiple nuclei (14N + 2x 1H)
fprintf('  Case 5: 14N + 2H...\n');

Sys5.S = 1/2;
Sys5.g = 2.006;
Sys5.Nucs = '14N,1H,1H';
Sys5.A = [16.0, 5.0, 5.0];  % MHz
Sys5.lw = [0.2, 0.0];

Exp5.mwFreq = 9.5;
Exp5.Range = [330, 348];
Exp5.nPoints = 1024;
Exp5.Harmonic = 1;

[B5, spec5] = garlic(Sys5, Exp5);

ref5.B = B5;
ref5.spec = spec5;
ref5.Sys = Sys5;
ref5.Exp = Exp5;
ref5.description = '14N + 2x 1H, multiple nuclei';

%% Test Case 6: Anisotropic A (averaged)
fprintf('  Case 6: Anisotropic A...\n');

Sys6.S = 1/2;
Sys6.g = 2.006;
Sys6.Nucs = '14N';
Sys6.A = [20, 20, 85];  % MHz, anisotropic (typical nitroxide)
Sys6.lw = [0.3, 0.0];

Exp6.mwFreq = 9.5;
Exp6.Range = [330, 348];
Exp6.nPoints = 512;
Exp6.Harmonic = 1;

[B6, spec6] = garlic(Sys6, Exp6);

ref6.B = B6;
ref6.spec = spec6;
ref6.Sys = Sys6;
ref6.Exp = Exp6;
ref6.description = 'Anisotropic A (nitroxide), averaged';

%% Test Case 7: Second derivative
fprintf('  Case 7: Second derivative...\n');

Sys7.S = 1/2;
Sys7.g = 2.006;
Sys7.Nucs = '14N';
Sys7.A = 16.0;
Sys7.lw = [0.3, 0.0];

Exp7.mwFreq = 9.5;
Exp7.Range = [333, 345];
Exp7.nPoints = 512;
Exp7.Harmonic = 2;  % Second derivative

[B7, spec7] = garlic(Sys7, Exp7);

ref7.B = B7;
ref7.spec = spec7;
ref7.Sys = Sys7;
ref7.Exp = Exp7;
ref7.description = '14N, second derivative';

%% Test Case 8: Lorentzian broadening
fprintf('  Case 8: Lorentzian broadening...\n');

Sys8.S = 1/2;
Sys8.g = 2.006;
Sys8.Nucs = '14N';
Sys8.A = 16.0;
Sys8.lw = [0.0, 0.5];  % Lorentzian only

Exp8.mwFreq = 9.5;
Exp8.Range = [333, 345];
Exp8.nPoints = 512;
Exp8.Harmonic = 1;

[B8, spec8] = garlic(Sys8, Exp8);

ref8.B = B8;
ref8.spec = spec8;
ref8.Sys = Sys8;
ref8.Exp = Exp8;
ref8.description = '14N, Lorentzian broadening';

%% Save all reference data
fprintf('Saving to ref_garlic.mat...\n');

% Get script directory and construct absolute path
script_dir = fileparts(mfilename('fullpath'));
output_file = fullfile(script_dir, 'ref_garlic.mat');

save(output_file, ...
    'ref1', 'ref2', 'ref3', 'ref4', 'ref5', 'ref6', 'ref7', 'ref8', ...
    '-v7');  % Use v7 format instead of v7.3 for scipy.io.loadmat compatibility

fprintf('Done! Generated 8 reference cases.\n');
fprintf('File: %s\n', output_file);
