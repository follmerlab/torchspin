% Powder grids of the D3d/D4h/D6h/Oh/C6h/S6 families (B36): EasySpin's grids
% cover phi = 0..maxPhi with maxPhi from private/gridparam.m (D4h/Oh pi/4,
% D3d pi/3, C6h pi/3, S6 2pi/3). Field sweeps of high-spin systems with cubic
% and trigonal crystal-field terms whose hamsymm group selects these grids,
% with automatic and explicit Opt.GridSymmetry. Reference for
% torchspin/tests/test_sphgrid_maxphi_matlab_validation.py.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc,'hamsymm',hamsymm(Sys));
end
O = struct('GridSize',[19 4]);
Od4h = O; Od4h.GridSymmetry = 'D4h';
Od3d = O; Od3d.GridSymmetry = 'D3d';
% S = 5/2 with D and a tetragonal B4 (hamsymm D4h); [200 480] has orientations without resonances
Sd4h = struct('S',5/2,'g',2,'D',300,'lw',[1 0]); Sd4h.B4 = [30 0 0 0 10 0 0 0 0];
Ew = struct('mwFreq',9.5,'Range',[200 480],'Harmonic',1,'nPoints',1024);
cases{end+1} = mk('d4h_auto_wide',Sd4h,Ew,O);
cases{end+1} = mk('d4h_explicit_wide',Sd4h,Ew,Od4h);
% same on a D2h grid: with missing knots EasySpin interpolates every transition
% linearly (L1), also those resonant at all knots
Od2h = O; Od2h.GridSymmetry = 'D2h';
cases{end+1} = mk('d4h_d2hgrid_wide',Sd4h,Ew,Od2h);
% S = 5/2 with D and a trigonal B4 (hamsymm D3d); [0 800] has resonances at all knots
Sd3d = struct('S',5/2,'g',2,'D',300,'lw',[1 0]); Sd3d.B4 = [0 30 0 0 10 0 0 0 0];
Ef = struct('mwFreq',9.5,'Range',[0 800],'Harmonic',1,'nPoints',4096);
cases{end+1} = mk('d3d_auto_full',Sd3d,Ef,O);
cases{end+1} = mk('d3d_explicit_full',Sd3d,Ef,Od3d);
% S = 5/2 with a cubic B4 only (hamsymm Oh)
Soh = struct('S',5/2,'g',2,'lw',[1 0]); Soh.B4 = [150 0 0 0 30 0 0 0 0];
cases{end+1} = mk('oh_auto',Soh,struct('mwFreq',9.5,'Range',[250 430],'Harmonic',1,'nPoints',1024),O);
% S = 7/2 with a hexagonal B6 (hamsymm D6h), automatic and explicit C6h grid;
% the D3d system on an explicit S6 grid (open-phi grids; a lower-symmetry
% grid on a higher-symmetry system gives the same spectrum in EasySpin)
Sd6h = struct('S',7/2,'g',2,'D',300,'lw',[1 0]); Sd6h.B6 = [0.5 0 0 0 0 0 0.1 0 0 0 0 0 0.3];
Oc6h = O; Oc6h.GridSymmetry = 'C6h';
Os6 = O; Os6.GridSymmetry = 'S6';
cases{end+1} = mk('d6h_auto_full',Sd6h,Ef,O);
cases{end+1} = mk('d6h_c6hgrid_full',Sd6h,Ef,Oc6h);
cases{end+1} = mk('d3d_s6grid_full',Sd3d,Ef,Os6);
% the grids themselves (sphgrid, GridSize 7) for all 14 grid symmetries
grids = {};
for sym = {'O3','Dinfh','D6h','D4h','Oh','D3d','Th','D2h','C4h','C6h','C2h','S6','Ci','C1'}
  g = sphgrid(sym{1},7);
  grids{end+1} = struct('sym',sym{1},'maxPhi',g.maxPhi,'phi',g.phi,'theta',g.theta,'weights',g.weights);
end
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_sphgrid_maxphi.mat'),'cases','grids','-v7'); fprintf('Done: %d cases, %d grids\n',numel(cases),numel(grids));
