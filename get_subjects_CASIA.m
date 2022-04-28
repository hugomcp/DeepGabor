function ret=get_subjects_CASIA(files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 22-12-2021.
% Creates a ID for each subject in the dataset
% "Right" eyes have even IDs, "left" eyes have odd IDs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ret=zeros(numel(files),1);

for i=1:numel(files)
    if (iscell(files))
        ret(i) = str2double(files{i}(2:5)) * 2 + (files{i}(6) == 'L');
    else        
        ret(i) = str2double(files(i).name(2:5)) * 2 + (files(i).name(6) == 'L');
    end
end