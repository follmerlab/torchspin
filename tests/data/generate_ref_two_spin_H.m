% generate_ref_two_spin_H.m
% Generate MATLAB reference .mat file for a two-spin Heisenberg Hamiltonian
% for S=1/2, J=1.0 and save as tests/data/ref_two_spin_H_J1.mat

addpath(genpath(fileparts(mfilename('fullpath'))));

S = 0.5;
J = 1.0;

% single-spin operators
SX = sop(S,'x');
SY = sop(S,'y');
SZ = sop(S,'z');

% two-spin operators via kron
S1x = kron(SX, eye(size(SX)));
S1y = kron(SY, eye(size(SY)));
S1z = kron(SZ, eye(size(SZ)));

S2x = kron(eye(size(SX)), SX);
S2y = kron(eye(size(SY)), SY);
S2z = kron(eye(size(SZ)), SZ);

H = J * (S1x * S2x + S1y * S2y + S1z * S2z);

fname = fullfile(fileparts(mfilename('fullpath')), 'ref_two_spin_H_J1.mat');
save(fname, 'H');
fprintf('Wrote %s\n', fname);
