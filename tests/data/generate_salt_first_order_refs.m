% Generate EasySpin references for the absolute intensity scale of salt's
% first-order perturbation path (torchspin bug B16, fixed in 0.8.1).
% Run in MATLAB with EasySpin on the path from the tests/data/ directory.
%
%   >> cd tests/data
%   >> generate_salt_first_order_refs
%
% Saves salt_first_order_refs.mat with cell array `ref`, one struct per case:
%   .Sys, .Exp, .Opt  inputs (fixed-field salt)
%   .x                rf axis (MHz)
%   .ymatrix          salt with Opt.Method = 'matrix' (EasySpin default)
%   .yperturb1        salt with Opt.Method = 'perturb1'
% Both methods give line intensities |<i|mu_perp|j>|^2 in (MHz/mT)^2.

clear ref
ref = {};

% 1) one 1H, rhombic g (the salt_1H_rhombicg system)
Sys = struct('S',1/2,'g',[2.0 2.01 2.02],'Nucs','1H','A',[-2 -2 4],'lwEndor',0.1);
Exp = struct('Field',326.5,'Range',[10 20],'nPoints',1001);
Opt = struct('GridSize',[31 4]);
ref{end+1} = run_case(Sys,Exp,Opt);

% 2) 1H + 14N without Q (different gn and I: relative weights matter)
Sys = struct('S',1/2,'g',[2.0 2.01 2.02],'Nucs','1H,14N', ...
  'A',[-2 -2 4; 5 5 7],'lwEndor',0.1);
Exp = struct('Field',326.5,'Range',[0 25],'nPoints',2001);
Opt = struct('GridSize',[31 4]);
ref{end+1} = run_case(Sys,Exp,Opt);

save('salt_first_order_refs.mat','ref','-v7');
fprintf('saved salt_first_order_refs.mat (%d cases)\n',numel(ref));

function r = run_case(Sys,Exp,Opt)
  Opt.Method = 'matrix';
  [x,ym] = salt(Sys,Exp,Opt);
  Opt.Method = 'perturb1';
  [~,yp] = salt(Sys,Exp,Opt);
  Opt = rmfield(Opt,'Method');
  r = struct('Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x(:)','ymatrix',ym(:)','yperturb1',yp(:)');
end
