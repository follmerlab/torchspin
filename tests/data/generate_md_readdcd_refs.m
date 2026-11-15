% EasySpin md_readdcd oracle for the bundled DCD trajectories in tests/mdfiles
% (torchspin 0.9.34, bug B97: CHARMM unit-cell block skipped from the right header field).
% Stores, per file, the coordinates of a fixed atom subset (1-based idx; the
% first 50 atoms, every 97th atom and the last 50), nAtoms, nFrames and dt.
% md_readdcd is in easyspin/private, so it is called from inside that folder.
clear all
script_dir = fileparts(mfilename('fullpath'));
mddir = fullfile(fileparts(script_dir),'mdfiles');
privdir = fullfile(fileparts(which('easyspin')),'private');
names = {'A10R1_polyAla','A10TOAC_polyAla'};
olddir = cd(privdir); restore = onCleanup(@() cd(olddir));
for k = 1:numel(names)
  fname = fullfile(mddir,[names{k} '.dcd']);
  full = md_readdcd(fname);
  N = full.nAtoms;
  idx = unique([1:50, 1:97:N, N-49:N]);
  T = md_readdcd(fname, idx);
  s = struct();
  s.idx = idx;                 % 1-based atom indices
  s.xyz = T.xyz;               % (nFrames, 3, numel(idx)), Angstrom
  s.nAtoms = T.nAtoms;
  s.nFrames = T.nFrames;
  s.dt = T.dt;
  s.xyz_full_frame1 = squeeze(full.xyz(1,:,:));  % (3, nAtoms), first frame, all atoms
  refs.(names{k}) = s; %#ok
  fprintf('%s: nAtoms %d, nFrames %d, dt %g, first x %g\n', names{k}, N, T.nFrames, T.dt, T.xyz(1,1,1));
end
clear restore
save(fullfile(script_dir,'ref_md_readdcd.mat'),'-struct','refs','-v7');
