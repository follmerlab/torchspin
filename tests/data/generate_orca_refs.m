% EasySpin orca2easyspin oracle for the ORCA files shipped with torchspin and EasySpin.
clear all
here = fileparts(mfilename('fullpath')); dirs = {fullfile(fileparts(here),'orca'), fullfile(fileparts(fileparts(fileparts(here))),'EasySpin','tests','orca')};
entries = {};
for d = 1:numel(dirs)
  files = dir(fullfile(dirs{d},'*'));
  for k = 1:numel(files)
    if files(k).isdir, continue; end
    [~,base,ext] = fileparts(files(k).name);
    if ~any(strcmpi(ext,{'.oof','.prop','.txt'})), continue; end
    fname = fullfile(files(k).folder, files(k).name);
    try
      Sys = orca2easyspin(fname);
    catch err
      fprintf('skip %s: %s\n', files(k).name, err.message); continue;
    end
    e = struct('file', [num2str(d) '/' files(k).name], 'n', numel(Sys));
    fn = fieldnames(Sys);
    for s = 1:numel(Sys)
      for f = 1:numel(fn)
        v = Sys(s).(fn{f});
        if isnumeric(v) || islogical(v) || ischar(v)
          e.(sprintf('s%d_%s', s, fn{f})) = v;
        end
      end
    end
    entries{end+1} = e; %#ok
    fprintf('ok %s (%d structures): %s\n', files(k).name, numel(Sys), strjoin(fn', ' '));
  end
end
save(fullfile(here,'ref_orca2easyspin.mat'),'entries','-v7'); fprintf('Done: %d files\n', numel(entries));
