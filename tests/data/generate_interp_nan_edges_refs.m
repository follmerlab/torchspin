% Grid interpolation with knots without a resonance (B48, B50): EasySpin's
% gridinterp 'L1' (private/gridinterp.m:129, interp2 'linear') ignores corners
% with zero weight, so a missing coarse knot does not propagate to fine knots on
% a coarse row or column, and sphgrid's phi is linspace(0,maxPhi,nPhi)
% (sphgrid.m:298,307). Field sweeps whose range cuts the powder pattern, the
% sphgrid knots of all open/closed-phi groups, and gridinterp 'L1' on synthetic
% data with missing knots. Reference for
% torchspin/tests/test_interp_nan_edges_matlab_validation.py.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
% B48: S = 1/2 rhombic g, range [325 337] mT cuts the g = 2.1 edge
Sys = struct('S',1/2,'g',[2 2.05 2.1]);
Exp = struct('mwFreq',9.5,'Range',[325 337],'nPoints',601,'Harmonic',0);
cases{end+1} = mk('b48_stick',Sys,Exp,struct('GridSize',[31 4]));
cases{end+1} = mk('b48_stick_noint',Sys,Exp,struct('GridSize',[31 1]));
Sys.lw = 1.5;
cases{end+1} = mk('b48_h0',Sys,Exp,struct('GridSize',[31 4]));
Exp.Harmonic = 1;
cases{end+1} = mk('b48_h1',Sys,Exp,struct('GridSize',[31 4]));
% B50: S = 5/2, D and tetragonal B4 on a D2h grid, range [290 390] mT
Sys = struct('S',5/2,'g',2,'D',300,'lw',[1 0]); Sys.B4 = [30 0 0 0 10 0 0 0 0];
Exp = struct('mwFreq',9.5,'Range',[290 390],'nPoints',1024,'Harmonic',1);
Opt = struct('GridSize',[19 4],'GridSymmetry','D2h');
cases{end+1} = mk('b50_h1',Sys,Exp,Opt);
Sys.lw = 0; Exp.Harmonic = 0;
cases{end+1} = mk('b50_stick',Sys,Exp,Opt);
Exp.Range = [200 480]; Exp.nPoints = 2801;
cases{end+1} = mk('b50_stick_wide',Sys,Exp,Opt);
% sphgrid knots (phi bitwise)
grids = {};
for sym = {'D2h','Ci','C1','C2h','C4h','C6h','S6','D4h','D6h','Oh','D3d','Th'}
  for N = [7 73]
    g = sphgrid(sym{1},N);
    grids{end+1} = struct('sym',sym{1},'N',N,'phi',g.phi,'theta',g.theta);
  end
end
% gridinterp 'L1' on synthetic data with missing knots
esdir = fileparts(which('pepper'));
old = cd(fullfile(esdir,'private')); gi = @gridinterp; cd(old);
interp = {};
spec = {'D2h',6,5; 'D2h',9,6; 'D2h',10,5; 'D2h',19,4; 'Ci',7,4; 'C1',6,4; 'C2h',8,3; 'D4h',7,4; 'C6h',6,4};
for k = 1:size(spec,1)
  sym = spec{k,1}; Nc = spec{k,2}; f = spec{k,3}; Nf = (Nc-1)*f+1;
  gc = sphgrid(sym,Nc); gf = sphgrid(sym,Nf);
  n = numel(gc.phi);
  y = sin(1:n) + 2;
  y(mod(1:n,5)==2) = NaN;
  gp = [n,Nc,gc.closedPhi,gc.nOctants,gc.maxPhi];
  yi = gi(y,gp,gf.phi,gf.theta,'L1');
  interp{end+1} = struct('sym',sym,'Nc',Nc,'Nf',Nf,'y',y,'yi',yi);
end
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_interp_nan_edges.mat'),'cases','grids','interp','-v7');
fprintf('Done: %d cases, %d grids, %d interp\n',numel(cases),numel(grids),numel(interp));
