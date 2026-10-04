% Generate MATLAB EasySpin reference spectra for the garlic Breit-Rabi /
% perturbation / equivalent-nuclei / field-modulation / frequency-sweep port.
% Cases mirror EasySpin's own tests: garlic_perturb, garlic_manynuclei,
% garlic_modamp, garlic_freqsweep_exact, garlic_freqsweep_pt2, garlic_hfsign,
% garlic_equivnuclei, garlic_temperature, garlic_dispersion1.
% Run from tests/data with EasySpin on the path:
%   matlab -batch "cd('tests/data'); addpath(genpath('<EasySpin>/easyspin')); generate_garlic_exact_refs"

clear all
fprintf('Generating garlic exact/perturb reference data...\n');
cases = {};

% Case: garlic_perturb -- 3 equivalent 1H, A = 600 MHz, all methods
Sys = struct('g',2,'Nucs','1H','A',600,'n',3,'lw',[0 0.1]);
Exp = struct('mwFreq',9.7,'nPoints',10000,'Range',[310 380]);
methods = {'perturb1','perturb2','perturb3','perturb4','perturb5','exact'};
for k = 1:numel(methods)
  Opt = struct('Method',methods{k});
  [x,y,info] = garlic(Sys,Exp,Opt);
  c = struct('name',['perturb_' methods{k}],'Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'y',y,'resfields',info.resfields);
  cases{end+1} = c; %#ok
end

% Case: garlic_manynuclei -- 1H n=5 + 14N n=4, autorange (record the range)
Sys = struct('g',2,'Nucs','1H,14N','A',[30,40],'n',[5 4],'lw',[0 0.1]);
Exp = struct('mwFreq',9.7,'nPoints',2000);
[x,y,info] = garlic(Sys,Exp);
Exp.Range = [x(1) x(end)];
cases{end+1} = struct('name','manynuclei','Sys',Sys,'Exp',Exp,'Opt',struct(),'x',x,'y',y,'resfields',info.resfields);

% Case: garlic_modamp -- field modulation, ModAmp 5 mT, lwpp 0.5
Sys = struct('lwpp',0.5);
Exp = struct('mwFreq',9.7,'CenterSweep',[346 10],'ModAmp',5);
[x,y] = garlic(Sys,Exp);
cases{end+1} = struct('name','modamp','Sys',Sys,'Exp',Exp,'Opt',struct(),'x',x,'y',y);

% Case: modamp with hyperfine, 2nd harmonic
Sys = struct('g',2.005,'Nucs','14N','A',45,'lwpp',[0 0.3]);
Exp = struct('mwFreq',9.7,'CenterSweep',[345.5 10],'ModAmp',1.0,'Harmonic',2,'nPoints',2048);
[x,y] = garlic(Sys,Exp);
cases{end+1} = struct('name','modamp_h2','Sys',Sys,'Exp',Exp,'Opt',struct(),'x',x,'y',y);

% Case: garlic_freqsweep_exact
Sys = struct('g',2,'Nucs','1H','A',200,'lw',10);
Exp = struct('Harmonic',0,'Field',330,'mwRange',[9.1 9.4]);
[x,y] = garlic(Sys,Exp);
cases{end+1} = struct('name','freqsweep_exact','Sys',Sys,'Exp',Exp,'Opt',struct(),'x',x,'y',y);

% Case: frequency sweep, perturb2, Lorentzian, harmonic 1, 14N + 1H
Sys = struct('g',2.003,'Nucs','14N,1H','A',[40 12],'lw',[0 3]);
Exp = struct('Harmonic',1,'Field',350,'mwRange',[9.6 10.0],'nPoints',4096);
Opt = struct('Method','perturb2');
[x,y] = garlic(Sys,Exp,Opt);
cases{end+1} = struct('name','freqsweep_pt2','Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'y',y);

% Case: garlic_hfsign -- negative vs positive A must be identical (field sweep)
Sys = struct('Nucs','1H','lwpp',0.1);
Exp = struct('Range',[346 354],'mwFreq',9.81,'nPoints',4096);
Opt = struct('Method','exact');
Sys.A = -100; [x,yneg] = garlic(Sys,Exp,Opt);
Sys.A = +100; [x,ypos] = garlic(Sys,Exp,Opt);
cases{end+1} = struct('name','hfsign','Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'y',ypos,'yneg',yneg);

% Case: garlic_equivnuclei -- n=4 vs four separate 1H (perturb2)
Exp = struct('mwFreq',9.669,'CenterSweep',[345 5],'Harmonic',0,'nPoints',10000);
Opt = struct('Method','perturb2');
Sys1 = struct('n',4,'lw',[0 0.02],'Nucs','1H','A',10);
Sys2 = struct('lw',[0 0.02],'Nucs','1H,1H,1H,1H','A',[10 10 10 10]);
[x,y1] = garlic(Sys1,Exp,Opt);
[x,y2] = garlic(Sys2,Exp,Opt);
cases{end+1} = struct('name','equivnuclei','Sys',Sys1,'Sys2',Sys2,'Exp',Exp,'Opt',Opt,'x',x,'y',y1,'y2',y2);

% Case: garlic_temperature -- Boltzmann polarization scales the spectrum
Sys = struct('g',2.0,'Nucs','14N','A',30,'lw',[0.2 0]);
Exp0 = struct('mwFreq',9.5,'Range',[335 343],'nPoints',1024);
ExpT = Exp0; ExpT.Temperature = 0.5;
[x,y0] = garlic(Sys,Exp0);
[x,yT] = garlic(Sys,ExpT);
cases{end+1} = struct('name','temperature','Sys',Sys,'Exp',ExpT,'Opt',struct(),'x',x,'y',yT,'y0',y0);

% Case: large hyperfine, 63Cu, exact vs perturb2 (second-order shifts matter)
Sys = struct('g',2.05,'Nucs','63Cu','A',150,'lw',[0 1]);
Exp = struct('mwFreq',9.5,'Range',[300 360],'nPoints',4096);
[x,ye,info] = garlic(Sys,Exp,struct('Method','exact'));
[x,yp] = garlic(Sys,Exp,struct('Method','perturb2'));
cases{end+1} = struct('name','cu_exact','Sys',Sys,'Exp',Exp,'Opt',struct('Method','exact'),'x',x,'y',ye,'y_pt2',yp,'resfields',info.resfields);

% Case: full A matrix + full g matrix (isotropic parts extracted via eig)
Sys = struct('g',[2.0 0.01 0; 0.01 2.01 0; 0 0 2.02],'Nucs','1H','lw',[0 0.05]);
Sys.A = [30 5 0; 5 28 0; 0 0 32];
Exp = struct('mwFreq',9.5,'CenterSweep',[338.5 4],'nPoints',2048);
[x,y,info] = garlic(Sys,Exp);
cases{end+1} = struct('name','fullAg','Sys',Sys,'Exp',Exp,'Opt',struct(),'x',x,'y',y,'resfields',info.resfields);

% Case: fast-motion + explicit accumulation + Gaussian residual (perturb5)
Sys = struct('g',[2.0088 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'tcorr',5e-10,'lw',[0.05 0.02]);
Exp = struct('mwFreq',9.5,'CenterSweep',[338.5 8],'nPoints',2048);
Opt = struct('Method','perturb5');
[x,y] = garlic(Sys,Exp,Opt);
cases{end+1} = struct('name','fastmotion_pt5','Sys',Sys,'Exp',Exp,'Opt',Opt,'x',x,'y',y);

% Case: fast-motion + modulation
Exp = struct('mwFreq',9.5,'CenterSweep',[338.5 8],'nPoints',2048,'ModAmp',0.2);
[x,y] = garlic(Sys,Exp);
cases{end+1} = struct('name','fastmotion_modamp','Sys',Sys,'Exp',Exp,'Opt',struct(),'x',x,'y',y);

script_dir = fileparts(mfilename('fullpath'));
save(fullfile(script_dir,'ref_garlic_exact.mat'),'cases','-v7');
fprintf('Done: %d cases -> ref_garlic_exact.mat\n',numel(cases));
