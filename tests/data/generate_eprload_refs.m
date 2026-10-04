% EasySpin eprload oracle for every file in tests/eprfiles (PORT_SPEC_2 item 4.5).
clear all
root = fullfile(fileparts(fileparts(mfilename('fullpath'))),'eprfiles');
files = dir(fullfile(root,'**','*')); files = files(~[files.isdir]);
entries = {};
for k = 1:numel(files)
  fname = fullfile(files(k).folder, files(k).name);
  [~,~,ext] = fileparts(files(k).name);
  if any(strcmpi(ext,{'.par','.dsc','.exp','.txt','.md'})), continue; end  % companion/parameter files
  try
    [x,y,pars] = eprload(fname);
  catch err
    fprintf('skip %s: %s\n', files(k).name, err.message); continue;
  end
  e = struct(); e.file = strrep(fname,[root filesep],''); 
  if iscell(x), e.x = x; e.xcell = true; else, e.x = x; e.xcell = false; end
  e.y = y; e.iscomplex = ~isreal(y);
  entries{end+1} = e; %#ok
  fprintf('ok %s: size %s\n', e.file, mat2str(size(y)));
end
script_dir = fileparts(mfilename('fullpath')); save(fullfile(script_dir,'ref_eprload.mat'),'entries','-v7');
fprintf('Done: %d files\n', numel(entries));
