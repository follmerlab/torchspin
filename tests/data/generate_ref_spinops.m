% generate_ref_spinops.m
% Generate MATLAB reference .mat files for spin operator tests used by the Python
% migration scaffold. Saves SX, SY, SZ for S=0.5 and S=1 to tests/data.

% Ensure repository functions on path
addpath(genpath(fileparts(mfilename('fullpath'))));

Slist = [0.5, 1];
for k=1:numel(Slist)
  S = Slist(k);
  SX = sop(S,'x');
  SY = sop(S,'y');
  SZ = sop(S,'z');
  fname = fullfile(fileparts(mfilename('fullpath')), sprintf('ref_spinops_S%g.mat', S));
  save(fname, 'SX', 'SY', 'SZ');
  fprintf('Wrote %s\n', fname);
end
