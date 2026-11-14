% Resonance transitions drawn by levelsplot with mwFreq (B78, 0.9.19).
% levelsplot.m:244-272 calls resfields(Sys,Exp,struct('Threshold',0,'Freq2Field',0))
% with Exp.SampleFrame = [-chi -theta -phi] and draws one line per transition
% between the two levels at the resonance field (Tag 'transition', UserData
% [level1 level2 B absintensity]). Stored per case: the resfields output and
% the drawn transition lines, plus resfields with rediagonalization at every
% resonance field. Reference for
% torchspin/tests/test_levelsplot_resonances.py.
clear all; cases = {};
function c = mk(name,Sys,Ori,B,mwFreq)
  if ischar(Ori)
    n = double([any(Ori=='x') any(Ori=='y') any(Ori=='z')]); [phi,theta] = vec2ang(n(:)); chi = 0;  % letter2vec is private; only letters without signs are used here
  elseif numel(Ori)==2
    phi = Ori(1); theta = Ori(2); chi = 0;
  else
    phi = Ori(1); theta = Ori(2); chi = Ori(3);
  end
  Exp = struct('mwFreq',mwFreq,'Range',B([1 end]));
  Exp.SampleFrame = [-chi -theta -phi];
  [Bres,Int,~,Tr] = resfields(Sys,Exp,struct('Threshold',0,'Freq2Field',0));
  % Same with rediagonalization at every resonance field (RediagLimit > 1): by
  % default resfields interpolates the eigenvectors linearly between segment
  % knots when they change little (RediagLimit 0.95, resfields.m:1116-1160),
  % which shifts its intensities by up to ~1e-3 relative.
  [BresX,IntX,~,TrX] = resfields(Sys,Exp,struct('Threshold',0,'Freq2Field',0,'RediagLimit',2));
  f = figure('Visible','off');
  levelsplot(Sys,Ori,B,mwFreq);
  h = findobj(f,'Tag','transition');
  ud = zeros(numel(h),4); Y = zeros(numel(h),2); X = zeros(numel(h),1); col = zeros(numel(h),3);
  for k = 1:numel(h)
    ud(k,:) = h(k).UserData; Y(k,:) = h(k).YData; X(k) = h(k).XData(1); col(k,:) = h(k).Color;
  end
  close(f);
  c = struct('name',name,'Sys',Sys,'Ori',Ori,'B',B,'mwFreq',mwFreq,'Bres',Bres(:),'Int',Int(:),'Tr',Tr, ...
    'BresRediag',BresX(:),'IntRediag',IntX(:),'TrRediag',TrX, ...
    'lineUserData',ud,'lineX',X,'lineY',Y,'lineColor',col);
end
cases{end+1} = mk('triplet_axial',struct('S',1,'g',2,'D',[3000 0]),[0.3 0.7 0.2],[0 600],9.5);
cases{end+1} = mk('triplet_rhombic',struct('S',1,'g',2,'D',[3000 600]),[1.1 0.4 0.9],[0 600],9.5);
cases{end+1} = mk('triplet_xy',struct('S',1,'g',2,'D',[3000 600]),'xy',[0 600],9.5);
cases{end+1} = mk('quartet',struct('S',3/2,'g',[2 2.05 2.1],'D',[5000 1000]),[0.5 1.2 2.0],[0 1400],9.5);
cases{end+1} = mk('doublet_14N',struct('S',1/2,'g',[2 2.01 2.02],'Nucs','14N','A',[20 30 90]),[0.4 0.6],[310 360],9.5);
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_levelsplot.mat'),'cases','-v7'); fprintf('Done: %d cases\n',numel(cases));
