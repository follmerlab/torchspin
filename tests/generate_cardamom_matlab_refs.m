% GENERATE_CARDAMOM_MATLAB_REFS  Produce MATLAB cardamom reference .mat files.
%
% Usage:
%   matlab -batch "addpath('easyspin'); run('tests/generate_cardamom_matlab_refs.m')"
%
% This script generates INDEPENDENT MATLAB reference spectra for
% cardamom cross-validation. Each call uses a fixed RNG seed so the
% output is bit-reproducible across runs. The Python companion tests in
% torchspin/tests/test_cardamom_matlab_validation.py load these files
% and compare via cosine similarity.
%
% Addresses audit finding P3B-C2 (prior references were Python-generated
% self-consistency data, not MATLAB oracles).

clear all;

outdir = fullfile(fileparts(mfilename('fullpath')), 'data');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% -----------------------------------------------------------------------
% Reference 1: Nitroxide, diffusion model, fast method, tcorr = 1 ns
% -----------------------------------------------------------------------
rng(42);

Sys.S = 1/2;
Sys.g = [2.008, 2.006, 2.003];
Sys.Nucs = '14N';
Sys.A = [20, 20, 85];
Sys.tcorr = 1e-9;

Exp.mwFreq = 9.5;
Exp.Range = [332, 352];
Exp.nPoints = 256;
Exp.Harmonic = 0;

Par.Model = 'diffusion';
Par.nTraj = 100;
Par.nSteps = 500;
Par.dtSpin = 1e-10;
Par.dtSpatial = 1e-10;

Opt.Method = 'fast';
Opt.Verbosity = 0;

[B, spc, TDSignal, t] = cardamom(Sys, Exp, Par, Opt);

meta.Sys = Sys;
meta.Exp = Exp;
meta.Par = Par;
meta.Opt = Opt;
meta.seed = 42;
meta.matlab_version = version;
meta.easyspin_version = easyspin('info');

save(fullfile(outdir, 'cardamom_matlab_diffusion_fast.mat'), ...
     'B', 'spc', 'TDSignal', 't', 'meta');

fprintf('Wrote: cardamom_matlab_diffusion_fast.mat\n');

% -----------------------------------------------------------------------
% Reference 2: Nitroxide, jump model (6 states), fast method
% -----------------------------------------------------------------------
rng(42);

clear Sys Exp Par Opt meta

Sys.S = 1/2;
Sys.g = [2.008, 2.006, 2.003];
Sys.Nucs = '14N';
Sys.A = [20, 20, 85];

Exp.mwFreq = 9.5;
Exp.Range = [332, 352];
Exp.nPoints = 256;
Exp.Harmonic = 0;

% Simple 2-state jump
Par.Model = 'jump';
Par.nTraj = 100;
Par.nSteps = 500;
Par.dtSpin = 1e-10;
Par.dtSpatial = 1e-10;
Sys.TransRates = [-1e9, 1e9; 1e9, -1e9];
Sys.Orientations = [0, 0, 0; pi/4, pi/4, 0]';

Opt.Method = 'fast';
Opt.Verbosity = 0;

[B, spc, TDSignal, t] = cardamom(Sys, Exp, Par, Opt);

meta.Sys = Sys;
meta.Exp = Exp;
meta.Par = Par;
meta.Opt = Opt;
meta.seed = 42;

save(fullfile(outdir, 'cardamom_matlab_jump_fast.mat'), ...
     'B', 'spc', 'TDSignal', 't', 'meta');

fprintf('Wrote: cardamom_matlab_jump_fast.mat\n');

% -----------------------------------------------------------------------
% Reference 3: Nitroxide, diffusion + ISTOs method
% -----------------------------------------------------------------------
rng(42);

clear Sys Exp Par Opt meta

Sys.S = 1/2;
Sys.g = [2.008, 2.006, 2.003];
Sys.Nucs = '14N';
Sys.A = [20, 20, 85];
Sys.tcorr = 1e-9;

Exp.mwFreq = 9.5;
Exp.Range = [332, 352];
Exp.nPoints = 256;
Exp.Harmonic = 0;

Par.Model = 'diffusion';
Par.nTraj = 50;
Par.nSteps = 1000;
Par.dtSpin = 1e-10;
Par.dtSpatial = 1e-10;

Opt.Method = 'ISTOs';
Opt.Verbosity = 0;

[B, spc, TDSignal, t] = cardamom(Sys, Exp, Par, Opt);

meta.Sys = Sys;
meta.Exp = Exp;
meta.Par = Par;
meta.Opt = Opt;
meta.seed = 42;

save(fullfile(outdir, 'cardamom_matlab_diffusion_istos.mat'), ...
     'B', 'spc', 'TDSignal', 't', 'meta');

fprintf('Wrote: cardamom_matlab_diffusion_istos.mat\n');

fprintf('\nAll cardamom MATLAB references generated successfully.\n');
fprintf('Run: python -m pytest torchspin/tests/test_cardamom_matlab_validation.py -v\n');
