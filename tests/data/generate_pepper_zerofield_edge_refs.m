% Zero-field lines exactly on bin edges with an interpolated powder grid
% (torchspin 0.8.14, pepper._zf_const_pos). S = 1, D = [-700 -1300 2000] MHz:
% the levels are at -2000, 700 and 1300 MHz, so the three lines (0.6, 2.7 and
% 3.3 GHz) coincide with points of the 10 MHz axis (0:0.01:5 GHz), i.e. with
% bin edges (bin k covers [x_k, x_k + dx)). GridSize [19 4] and [10 8]
% interpolate the coarse grid; EasySpin puts each line into one bin.
% For every grid and line width the full spectrum and the spectrum of each
% level pair (Opt.Transitions, 1-based, levels in ascending energy) are saved,
% so a test can rebuild the spectrum line by line when round-off on another
% machine puts a line into the neighbor bin. Gaussian widths only: a Gaussian
% line shifted by one bin equals the line of the shifted stick to round-off,
% a Lorentzian does not (its tails reach the ends of the axis; 1e-4).
% Saved to ref_pepper_zerofield_edge.mat (-v7).
clear all
fprintf('Generating pepper zero-field bin-edge references...\n');
Sys0 = struct('S',1,'g',[2 2 2],'D',[-700 -1300 2000]);
Exp = struct('Field',0,'mwRange',[0 5],'nPoints',501,'Harmonic',0);
grids = {[19 4],[10 8]};
lws = {[0 0],[30 0],[300 0]};   % Gaussian FWHM (MHz); [0 0]: sticks
pairs = [2 3; 1 2; 1 3];               % lines at 0.6, 2.7, 3.3 GHz
nu = []; spc = zeros(numel(grids),numel(lws),Exp.nPoints);
spc_line = zeros(numel(grids),numel(lws),size(pairs,1),Exp.nPoints);
for ig = 1:numel(grids)
  for il = 1:numel(lws)
    Sys = Sys0;
    if any(lws{il}), Sys.lw = lws{il}; end
    Opt = struct('GridSize',grids{ig},'Verbosity',0);
    [nu,y] = pepper(Sys,Exp,Opt);
    spc(ig,il,:) = y;
    for ip = 1:size(pairs,1)
      Opt.Transitions = pairs(ip,:);
      [~,y] = pepper(Sys,Exp,Opt);
      spc_line(ig,il,ip,:) = y;
    end
  end
end
GridSize = cell2mat(grids');
lw = cell2mat(lws');
D = Sys0.D;
script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_pepper_zerofield_edge.mat'), ...
     'nu','spc','spc_line','GridSize','lw','pairs','D','-v7');
fprintf('Done\n');
