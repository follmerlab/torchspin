% EasySpin orisel references (torchspin 0.9.0, bug B26: orisel redesigned to EasySpin's model).
% For every case: [W,Trans] = orisel(Sys,Exp,Opt) (per orientation and level pair),
% w = orisel(Sys,Exp,Opt) (summed over pairs), and the orientations [phi;theta] used.
% Cases with Exp.Orientations feed torchspin exactly the same orientations (isolates the
% weight model); grid cases (no Exp.Orientations) also store the grid orisel.m builds
% (orisel.m:79-93: hamsymm -> GridFrame, sphgrid, vec2ang(GridFrame*vecs)) so torchspin's
% grid and GridFrame can be checked orientation by orientation.
% Run: matlab -batch "addpath(genpath('<easyspin>')); cd tests/data; generate_orisel_refs"
clear all
fprintf('Generating orisel references...\n');

mw = 9.5;                                  % GHz
Bof = @(g) planck*mw*1e9/(g*bmagn)*1e3;    % resonance field (mT) of an effective g

% 400-point Fibonacci sphere, [phi;theta]
N = 400; k = 0:N-1;
fib = [mod(k*pi*(3-sqrt(5)),2*pi); acos(1-2*(k+0.5)/N)];

R = struct;
cs = {};

rh = struct('S',1/2,'g',[2.0 2.05 2.1]);
cs{end+1} = {'rh_gz', rh, struct('Field',Bof(2.1),'ExciteWidth',100,'Orientations',fib), struct};
cs{end+1} = {'rh_gy', rh, struct('Field',Bof(2.05),'ExciteWidth',100,'Orientations',fib), struct};
cs{end+1} = {'rh_gx', rh, struct('Field',Bof(2.0),'ExciteWidth',100,'Orientations',fib), struct};

cu = struct('S',1/2,'g',[2.05 2.05 2.25],'Nucs','63Cu','A',[50 50 500]);
cs{end+1} = {'cu_294', cu, struct('Field',294,'ExciteWidth',100,'Orientations',fib), struct};
cs{end+1} = {'cu_perp', cu, struct('Field',Bof(2.05),'ExciteWidth',100,'Orientations',fib), struct};

g6 = struct('S',1/2,'g',[1.9 2.0 6.0]);
cs{end+1} = {'g6_g6', g6, struct('Field',Bof(6.0),'ExciteWidth',100,'Orientations',fib), struct};
cs{end+1} = {'g6_g4', g6, struct('Field',Bof(4.0),'ExciteWidth',100,'Orientations',fib), struct};

s1 = struct('S',1,'g',2.0,'D',[1500 300],'HStrain',[60 80 100]);
cs{end+1} = {'s1_hstrain', s1, struct('Field',300,'ExciteWidth',200,'Orientations',fib), struct};
cs{end+1} = {'s1_noavg', s1, struct('Field',300,'ExciteWidth',200,'Orientations',fib), struct('AveragedIntensity',false)};

s32 = struct('S',3/2,'g',2.0,'D',[3000 600],'HStrain',[50 50 50]);
cs{end+1} = {'s32_geff4', s32, struct('Field',160,'ExciteWidth',150,'Orientations',fib), struct};

two = struct('S',[1/2 1/2],'g',[2.0 2.0 2.1; 2.05 2.05 2.0],'ee',[100 100 -200]);
cs{end+1} = {'two_electrons', two, struct('Field',330,'ExciteWidth',300,'Orientations',fib), struct};

cs{end+1} = {'inf_excitewidth', rh, struct('Field',330,'ExciteWidth',Inf,'Orientations',fib(:,1:7)), struct};

% default grids (GridSymmetry from hamsymm, grid in GridFrame)
rhs = struct('S',1/2,'g',[2.0 2.05 2.1],'HStrain',[30 40 60]);
cs{end+1} = {'grid_rh_default', rhs, struct('Field',Bof(2.05),'ExciteWidth',100), struct};
rot = struct('S',1/2,'g',[2.0 2.1 2.2],'gFrame',[10 30 0]*pi/180,'HStrain',[30 40 60]);
cs{end+1} = {'grid_rotated_d2h', rot, struct('Field',Bof(2.1),'ExciteWidth',100), struct('GridSize',20)};
cuf = struct('S',1/2,'g',[2.05 2.05 2.25],'Nucs','63Cu','A',[50 50 500],'AFrame',[0 20 0]*pi/180);
cs{end+1} = {'grid_cu_aframe', cuf, struct('Field',294,'ExciteWidth',100), struct('GridSize',15)};
ax = struct('S',1/2,'g',[2.0 2.0 2.2]);
cs{end+1} = {'grid_axial', ax, struct('Field',Bof(2.1),'ExciteWidth',100), struct('GridSize',30)};
cs{end+1} = {'grid_explicit_ci', rot, struct('Field',Bof(2.1),'ExciteWidth',100), struct('GridSize',10,'GridSymmetry','Ci')};
iso = struct('S',1/2,'g',2.0);
cs{end+1} = {'grid_isotropic', iso, struct('Field',Bof(2.0)+0.5,'ExciteWidth',100), struct};

for c = 1:numel(cs)
  name = cs{c}{1}; Sys = cs{c}{2}; Exp = cs{c}{3}; Opt = cs{c}{4};
  Exp.mwFreq = mw;
  [W,Trans] = orisel(Sys,Exp,Opt);
  w = orisel(Sys,Exp,Opt);
  if isfield(Exp,'Orientations')
    ori = Exp.Orientations;
  else
    % the grid orisel.m:79-93 builds
    if ~isfield(Opt,'GridSize'), Opt.GridSize = 46; end
    if ~isfield(Opt,'GridSymmetry') || isempty(Opt.GridSymmetry)
      [sym,GridFrame] = hamsymm(Sys);
    else
      sym = Opt.GridSymmetry; GridFrame = eye(3);
    end
    grid = sphgrid(sym,Opt.GridSize);
    [phi,theta] = vec2ang(GridFrame*grid.vecs);
    ori = [phi; theta];
    Exp.GridSymmetry = sym; Exp.GridFrame = GridFrame;
  end
  R.(name) = struct('Sys',Sys,'Exp',Exp,'Opt',Opt,'W',W,'Trans',Trans,'w',w,'ori',ori);
  fprintf('  %-18s nOri=%4d nTrans=%2d max(w)=%.4g\n',name,size(W,1),size(W,2),max(w));
end

save('ref_orisel.mat','-struct','R','-v7');
fprintf('saved ref_orisel.mat\n');
