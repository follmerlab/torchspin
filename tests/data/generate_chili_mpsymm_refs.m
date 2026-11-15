% chili references for the M-pS-pI symmetry / Lanczos fixes (torchspin B114, B116).
% Nitroxide in the slow-motion limit (tcorr = 100 ns, default LLMK [14 7 2 6]).
% Run ONE part per MATLAB job, from the repository root:
%   matlab -singleCompThread -batch "part='a'; run('tests/data/generate_chili_mpsymm_refs.m')"
%   matlab -singleCompThread -batch "part='b'; run('tests/data/generate_chili_mpsymm_refs.m')"
%   matlab -singleCompThread -batch "part='c'; run('tests/data/generate_chili_mpsymm_refs.m')"
%   matlab -singleCompThread -batch "part='d'; run('tests/data/generate_chili_mpsymm_refs.m')"
% Part a -> ref_chili_mpsymm_a.mat: fast builder (yf, EasySpin's default), general method with
%   its Lanczos solver (yg) and general method with the direct solver Solver='\' (yd).
% Part b -> ref_chili_mpsymm_b.mat: the same with Opt.highField = true: fast (yf), general
%   direct (yd).
% Part c -> ref_chili_mpsymm_c.mat: fast builder with a tighter Lanczos threshold,
%   Opt.Threshold = 1e-10 (yf10) and 1e-12 (yf12), to test whether the default fast output
%   (Threshold 1e-6) is a converged continued fraction.
% Part d -> ref_chili_mpsymm_d.mat: case simple_tc1e-07 of generate_chili_ext_refs.m (other
%   system, default nPoints) with the fast builder at the default Threshold (yf) and at
%   Threshold 1e-10 (yf10).
if ~exist('part','var'), error('Set part = ''a'', ''b'', ''c'' or ''d'' before running this script.'); end
Sys = struct('g',[2.0089 2.0058 2.0021],'Nucs','14N','A',[16 16 100],'tcorr',1e-7,'lw',[0 0.05]);
Exp = struct('mwFreq',9.5,'Range',[328 352],'nPoints',256,'Harmonic',1);
chili_path = which('chili');
script_dir = fileparts(mfilename('fullpath'));
switch part
  case 'a'
    [x,yf] = chili(Sys,Exp,struct('Verbosity',0));
    [~,yg] = chili(Sys,Exp,struct('Verbosity',0,'LiouvMethod','general'));
    [~,yd] = chili(Sys,Exp,struct('Verbosity',0,'LiouvMethod','general','Solver','\'));
    save(fullfile(script_dir,'ref_chili_mpsymm_a.mat'),'Sys','Exp','x','yf','yg','yd','chili_path','-v7');
  case 'b'
    [x,yf] = chili(Sys,Exp,struct('Verbosity',0,'highField',true));
    [~,yd] = chili(Sys,Exp,struct('Verbosity',0,'highField',true,'LiouvMethod','general','Solver','\'));
    save(fullfile(script_dir,'ref_chili_mpsymm_b.mat'),'Sys','Exp','x','yf','yd','chili_path','-v7');
  case 'c'
    [x,yf10] = chili(Sys,Exp,struct('Verbosity',0,'Threshold',1e-10));
    [~,yf12] = chili(Sys,Exp,struct('Verbosity',0,'Threshold',1e-12));
    save(fullfile(script_dir,'ref_chili_mpsymm_c.mat'),'Sys','Exp','x','yf10','yf12','chili_path','-v7');
  case 'd'
    Sys = struct('g',[2.008 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'lw',0.1,'tcorr',1e-7);
    Exp = struct('mwFreq',9.8,'Range',[343 358]);
    [x,yf] = chili(Sys,Exp,struct('Verbosity',0));
    [~,yf10] = chili(Sys,Exp,struct('Verbosity',0,'Threshold',1e-10));
    save(fullfile(script_dir,'ref_chili_mpsymm_d.mat'),'Sys','Exp','x','yf','yf10','chili_path','-v7');
  otherwise
    error('Unknown part ''%s''.',part);
end
c = @(a,b) (a(:)'*b(:))/norm(a)/norm(b);
r = @(a,b) max(abs(a))/max(abs(b));
switch part
  case 'a'
    fprintf('a: general-Lanczos vs fast: cos %.6f ratio %.5f\n',c(yg,yf),r(yg,yf));
    fprintf('a: general-direct  vs fast: cos %.6f ratio %.5f\n',c(yd,yf),r(yd,yf));
    fprintf('a: general-Lanczos vs general-direct: cos %.6f ratio %.5f\n',c(yg,yd),r(yg,yd));
  case 'b'
    fprintf('b: highField general-direct vs fast: cos %.6f ratio %.5f\n',c(yd,yf),r(yd,yf));
  case 'c'
    a = load(fullfile(script_dir,'ref_chili_mpsymm_a.mat'));
    fprintf('c: fast(1e-10) vs fast(1e-6): cos %.6f ratio %.5f; vs general-direct: cos %.6f ratio %.5f\n',c(yf10,a.yf),r(yf10,a.yf),c(yf10,a.yd),r(yf10,a.yd));
    fprintf('c: fast(1e-12) vs general-direct: cos %.6f ratio %.5f\n',c(yf12,a.yd),r(yf12,a.yd));
  case 'd'
    fprintf('d: fast(1e-10) vs fast(1e-6): cos %.6f ratio %.5f\n',c(yf10,yf),r(yf10,yf));
end
fprintf('chili: %s\n',chili_path);
