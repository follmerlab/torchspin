% Generate EasySpin references for ENDOR transition selection (torchspin 0.9.14,
% bugs B57/B41, B40): EasySpin endorfrq selects level pairs once, on the coarse
% grid sphgrid(Opt.TPSGridSymm, Opt.TPSGridSize) (endorfrq.m:258-341), with
% electronic level pairs always removed, independent of the simulation grid.
% Run in MATLAB with EasySpin on the path from the tests/data/ directory:
%
%   >> cd tests/data
%   >> generate_endor_tps_selection_refs
%
% Saves endor_tps_selection_refs.mat with struct `ref`:
%   ref.endorfrq{k}  single-orientation endorfrq (powder mode, as salt calls it):
%       .Sys, .Field, .phi, .theta, .Range ([] = none), .Threshold,
%       .TPSGridSymm, .TPSGridSize ('' / 0 = EasySpin default),
%       .Tr (1-based level pairs, EasySpin order), .Poff/.Ioff, .Pon/.Ion
%   ref.salt{k}      fixed-field salt spectra and selected transitions:
%       .Sys, .Exp, .Opt, .x, .yoff, .yon, .Troff, .Tron (info.Transitions)

clear ref
ref.endorfrq = {};
ref.salt = {};

sys1H = struct('S',1/2,'g',[2 2 2],'Nucs','1H','A',[3 7 12]);
sysNH = struct('S',1/2,'g',[2.0088 2.0061 2.0027],'Nucs','14N,1H', ...
  'A',[17 17 98; 2 3 5],'Q',[1.0 0.5 -1.5; 0 0 0]);
sysCuN = struct('S',1/2,'g',[2.05 2.06 2.25],'Nucs','63Cu,14N', ...
  'A',[60 70 550; 40 42 50],'Q',[-3 -3 6; 0.4 0.6 -1.0]);

% -------------------------------------------------------------------------
% endorfrq: [Sys, Field, Range, Threshold, TPSGridSymm, TPSGridSize]
% -------------------------------------------------------------------------
ecases = {
  sys1H,  350, [],      0,    '',    0     % B40: electronic pairs removed at threshold 0
  sysNH,  340, [],      1e-4, '',    0     % B41: default-threshold selection
  sysNH,  340, [0 20],  1e-4, '',    0     % range used by the selection only
  sysNH,  340, [],      0,    '',    0
  sysCuN, 300, [],      1e-4, '',    0
  sysNH,  340, [],      1e-2, 'Ci',  7     % Opt.TPSGridSymm / TPSGridSize
};
angles = [0 0; 0.3 0.7; 1.1 1.3];   % [phi theta] rows

for ic = 1:size(ecases,1)
  for ia = 1:size(angles,1)
    phi = angles(ia,1); theta = angles(ia,2);
    Exp = struct('Field',ecases{ic,2},'MolFrame',[0 -theta -phi]);
    if ~isempty(ecases{ic,3}), Exp.Range = ecases{ic,3}; end
    Opt = struct('crystalSample',false,'rotatedSample',false, ...
                 'Threshold',ecases{ic,4});
    if ~isempty(ecases{ic,5})
      Opt.TPSGridSymm = ecases{ic,5}; Opt.TPSGridSize = ecases{ic,6};
    end
    Opt.Enhancement = 'off';
    [Poff,Ioff,Tr] = endorfrq(ecases{ic,1},Exp,Opt);
    Opt.Enhancement = 'on';
    [Pon,Ion,Tr2] = endorfrq(ecases{ic,1},Exp,Opt);
    assert(isequal(Tr,Tr2));
    r = struct('Sys',ecases{ic,1},'Field',ecases{ic,2},'phi',phi,'theta',theta, ...
      'Range',ecases{ic,3},'Threshold',ecases{ic,4}, ...
      'TPSGridSymm',ecases{ic,5},'TPSGridSize',ecases{ic,6}, ...
      'Tr',Tr,'Poff',Poff,'Ioff',Ioff,'Pon',Pon,'Ion',Ion);
    ref.endorfrq{end+1} = r;
  end
end

% -------------------------------------------------------------------------
% salt (fixed field)
% -------------------------------------------------------------------------
% 14N + 1H (B57): zero-quantum pairs near 0 MHz; two grid sizes (the
% selection does not depend on GridSize)
Sys = sysNH; Sys.lwEndor = 0.3;
Exp = struct('Range',[0 70],'Field',340,'nPoints',2001,'mwFreq',9.55);
for gs = {[31 6],[13 4]}
  Opt = struct('Threshold',1e-4,'GridSize',gs{1});
  ref.salt{end+1} = run_salt(Sys,Exp,Opt);
end

% Cu(II) with large A and Q, rf axis up to 2 mwFreq (B40: electronic pairs
% near 8.9 GHz)
Sys = struct('S',1/2,'g',[2 2 2.2],'Nucs','63Cu','A',[60 60 600], ...
  'Q',[-5 -5 10],'lwEndor',30);
Exp = struct('Range',[0 19400],'Field',320,'nPoints',4001,'mwFreq',9.7);
Opt = struct('Threshold',1e-4,'GridSize',[19 4]);
ref.salt{end+1} = run_salt(Sys,Exp,Opt);

save('endor_tps_selection_refs.mat','ref','-v7');
fprintf('saved endor_tps_selection_refs.mat (%d endorfrq, %d salt cases)\n', ...
  numel(ref.endorfrq), numel(ref.salt));

function r = run_salt(Sys,Exp,Opt)
  Opt.Enhancement = 'off';
  [x,yoff,info] = salt(Sys,Exp,Opt);
  Troff = info.Transitions;
  Opt.Enhancement = 'on';
  [~,yon,info] = salt(Sys,Exp,Opt);
  Tron = info.Transitions;
  r = struct('Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x(:)','yon',yon(:)','yoff',yoff(:)', ...
    'Troff',Troff,'Tron',Tron);
end
