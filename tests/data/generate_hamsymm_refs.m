% generate_hamsymm_refs.m
% MATLAB EasySpin reference outputs for torchspin hamsymm validation.
%
% Saves, for each named case, the point group string and the symmetry-frame
% rotation matrix returned by EasySpin's hamsymm:
%     [PGroup, RMatrix] = hamsymm(Sys)
% (RMatrix columns = symmetry-frame axes in the molecular frame).
%
% Run from tests/data with EasySpin on the path:
%   /Applications/MATLAB_R2024b.app/bin/matlab -batch ...
%     "cd('<abs>/tests/data'); addpath(genpath('<EasySpin>/easyspin')); generate_hamsymm_refs"
%
% The spin systems are hard-coded identically in
% torchspin/tests/test_hamsymm_matlab_validation.py.

clear ref
fprintf('Generating hamsymm reference data...\n');

%% ---- Eigenvalue-branch cases (Stevens / full tensors / higher-order Zeeman)

% 1. cubic B4 term, S=7/2 -> Oh
B4 = 10;
Sys = struct('S',7/2,'g',[2 2 2]);
Sys.B4 = [5*B4 0 0 0 B4 0 0 0 0];
ref.cubic_B4 = runcase(Sys);

% 2. cubic B6 term -> Oh
B6 = 10;
Sys = struct('S',7/2,'g',[2 2 2]);
Sys.B6 = [0 0 -21*B6 0 0 0 B6 0 0 0 0 0 0];
ref.cubic_B6 = runcase(Sys);

% 3. cubic B4 + B6 -> Oh
B4 = 10; B6 = 12.87;
Sys = struct('S',7/2,'g',[2 2 2]);
Sys.B4 = [5*B4 0 0 0 B4 0 0 0 0];
Sys.B6 = [0 0 -21*B6 0 0 0 B6 0 0 0 0 0 0];
ref.cubic_B4B6 = runcase(Sys);

% 4. rhombic Stevens rank-2, S=1 -> D2h
b20 = 100; b22 = 150;
Sys = struct('S',1,'g',[2 2 2]);
Sys.B2 = [b22 0 b20 0 0];
ref.stevens_B2_rhombic = runcase(Sys);

% 5. axial B4^0 only, S=2 -> Dinfh
Sys = struct('S',2,'g',[2 2 2]);
Sys.B4 = [0 0 0 0 30 0 0 0 0];
ref.stevens_B40_axial = runcase(Sys);

% 6. tetragonal B4^0 + B4^4, S=2 -> D4h
Sys = struct('S',2,'g',[2 2 2]);
Sys.B4 = [3 0 0 0 30 0 0 0 0];
ref.stevens_B40_B44 = runcase(Sys);

% 7. trigonal B4^0 + B4^3, S=2 -> D3d
Sys = struct('S',2,'g',[2 2 2]);
Sys.B4 = [0 4 0 0 30 0 0 0 0];
ref.stevens_B40_B43 = runcase(Sys);

% 8. hexagonal B6^0 + B6^6, S=7/2 -> D6h
Sys = struct('S',7/2,'g',[2 2 2]);
Sys.B6 = [2 0 0 0 0 0 5 0 0 0 0 0 0];
ref.stevens_B60_B66 = runcase(Sys);

% 9. rhombic B2 with tilted B2Frame; the tilted frame is NOT among the
%    candidate frames of hamsymm_eigs (only tensor frames are tried)
b20 = 100; b22 = 150;
Sys = struct('S',1,'g',[2 2 2]);
Sys.B2 = [b22 0 b20 0 0];
Sys.B2Frame = [0.3 0.7 0.2];
ref.stevens_B2_tilted = runcase(Sys);

% 10. rhombic B2 with tilted B2Frame AND a g tensor in the same frame
%     -> the g frame is a candidate, so D2h should be recovered there
Sys = struct('S',1,'g',[2.0 2.1 2.2],'gFrame',[0.3 0.7 0.2]);
Sys.B2 = [b22 0 b20 0 0];
Sys.B2Frame = [0.3 0.7 0.2];
ref.stevens_B2_tilted_with_g = runcase(Sys);

% 11. full g matrix, axial along x (rotated principal axes) -> Dinfh via QM
Sys = struct('S',1/2);
Sys.g = diag([2.3 2.0 2.0]);
ref.fullg_axial_x = runcase(Sys);

% 12. full g matrix, rhombic and tilted (no tensor frame available)
R = erot([0.3 0.7 0.2]);
Sys = struct('S',1/2);
Sys.g = R.'*diag([2.0 2.1 2.2])*R;
ref.fullg_rhombic_tilted = runcase(Sys);

% 13. full A matrix, axial tilted by 24 deg about y (hamsymm_fullhf_magnitude)
Sys = struct('S',1/2,'Nucs','1H');
AFrame = [0 24 0]*pi/180;
R = erot(AFrame);
Sys.A = R*diag([1 1 2])*R.';
ref.fullA_axial_tilted = runcase(Sys);

% 14. full A matrix, scaled 1e-3 (magnitude independence)
Sys.A = Sys.A*1e-3;
ref.fullA_axial_tilted_small = runcase(Sys);

% 15. three S=1/2 with full antisymmetric ee matrices (hamsymm_fullee)
Sys = struct('S',[1/2 1/2 1/2],'g',[2 2 2]);
J1 = -100; J2 = -89.9; dz1 = 4.85; dz2 = -dz1;
ee12 = [-J1 dz1 0; -dz1 -J1 0; 0 0 -J1];
ee13 = [-J2 dz2 0; -dz2 -J2 0; 0 0 -J2];
Sys.ee = [ee12; ee13; ee12]*100*clight/1e6;
ref.fullee_threespins = runcase(Sys);

% 16. higher-order Zeeman term Ham132 (B^1 S^3, l=2), S=3/2, axial
Sys = struct('S',3/2,'g',[2 2 2]);
Sys.Ham132 = [0 0 0.05 0 0];
ref.ham132_axial = runcase(Sys);

%% ---- Geometric-branch cases (frame checks)

% 17. tilted rhombic g -> D2h, RMatrix = erot(gFrame).'
Sys = struct('S',1/2,'g',[2.0 2.1 2.2],'gFrame',[0.3 0.7 0.2]);
ref.geom_rhombic_tilted = runcase(Sys);

% 18. axial g, axial A tilted about y -> C2h
Sys = struct('S',1/2,'Nucs','1H','g',[2 2 3],'A',[3 3 1],'AFrame',[0 0.4 0]);
ref.geom_two_axial_ztilt = runcase(Sys);

% 19. rhombic g + axial A perpendicular (D2h + Dinfh, in a sigma plane) -> C2h
Sys = struct('S',1/2,'Nucs','1H','g',[2 2 3],'A',[3 2 1],'gFrame',[0.5 pi/2 0]);
ref.geom_rhombic_axial_plane = runcase(Sys);

% 20. two rhombic sharing one axis -> C2h
Sys = struct('S',1/2,'Nucs','1H','g',[2 2.5 3],'A',[3 2 1],'AFrame',[0.6 0 0]);
ref.geom_two_rhombic_one_axis = runcase(Sys);

% 21. axial g along x (principal-value permutation) -> Dinfh, x is the axis
Sys = struct('S',1/2,'g',[2.2 2 2]);
ref.geom_axial_x = runcase(Sys);

% 22. two electrons, axial hyperfine to both -> Dinfh (hamsymm_twospinA)
Sys = struct('S',[1/2 1/2],'g',[2 2],'Nucs','1H','A',[1 1 1 2 2 3],'ee',6);
ref.geom_twospinA = runcase(Sys);

%% ---- Save
script_dir = fileparts(mfilename('fullpath'));
outfile = fullfile(script_dir,'ref_hamsymm.mat');
save(outfile,'ref','-v7');
fprintf('Saved %d cases to %s\n',numel(fieldnames(ref)),outfile);

%% -------------------------------------------------------------------------
function out = runcase(Sys)
  [PGroup,RMatrix] = hamsymm(Sys);
  out.PGroup = PGroup;
  out.RMatrix = RMatrix;
  fprintf('  %-10s\n',PGroup);
end
