% Position interpolation on axial (Dinfh) grids (B30): EasySpin's G3 mode is
% esspline1d(y,1,iphi), a cubic spline over the theta knot index with zero
% end slopes (gridinterp.m:77-78, esspline1d.m EndCon=1). Reference for
% torchspin/tests/test_pepper_dinfh_interp_matlab_validation.py.
clear all; cases = {};
function c = mk(name,Sys,Exp,Opt)
  Opt.Verbosity = 0; [x,spc] = pepper(Sys,Exp,Opt);
  c = struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'spc',spc);
end
% axial g, coarse grid with a large interpolation factor
SysG = struct('S',1/2,'g',[2 2 2.4],'lw',[0.3 0]);
ExpG = struct('mwFreq',9.5,'Range',[270 350],'nPoints',4096,'Harmonic',1);
cases{end+1} = mk('g_axial_gs10x10_h1',SysG,ExpG,struct('GridSize',[10 10]));
cases{end+1} = mk('g_axial_default_h1',SysG,ExpG,struct());
ExpG0 = ExpG; ExpG0.Harmonic = 0;
cases{end+1} = mk('g_axial_default_h0',SysG,ExpG0,struct());
% Cu(II)-like axial g and 63Cu hyperfine
SysC = struct('S',1/2,'g',[2.05 2.05 2.25],'Nucs','63Cu','A',[50 500],'lw',[0.5 0]);
ExpC = struct('mwFreq',9.5,'Range',[250 360],'nPoints',4096,'Harmonic',1);
cases{end+1} = mk('cu_axial_default_h1',SysC,ExpC,struct());
ExpC0 = ExpC; ExpC0.Harmonic = 0;
cases{end+1} = mk('cu_axial_default_h0',SysC,ExpC0,struct());
cases{end+1} = mk('cu_axial_gs10x10_h1',SysC,ExpC,struct('GridSize',[10 10]));
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_pepper_dinfh_interp.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
