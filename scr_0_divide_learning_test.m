%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 22-12-2021.
% Divides a data set into Learn/Test (w/ disjoint subjects).
% Additionally, creates a subset of Learn, as Learn_Gabor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

DATASET_NAME = 'CASIA-Iris-Thousand';

FOLDER_IMGS=['/Users/hugoproenca/Desktop/',DATASET_NAME,'/segmented_normalized/'];

PROPORTION_SUBJECTS_TEST=0.05;

PROPORTION_SUBJECTS_LEARN_GABOR = 0.05;


files=dir([FOLDER_IMGS,'*_normalized_img.png']);
subjects_files = get_subjects_CASIA(files);
subjects_un=unique(subjects_files);
subjects_un=subjects_un(randperm(numel(subjects_un)));

tot_subj_test=round(numel(subjects_un)*PROPORTION_SUBJECTS_TEST);
tot=0;
data={};
for i=1:tot_subj_test
    aux=find(subjects_files==subjects_un(i));
    for j=1:numel(aux)
        data{end+1}=files(aux(j)).name;
    end    
    tot=tot+numel(aux);
end
writecell(data,['data/',DATASET_NAME,'_test.csv'],'Delimiter',',');
fprintf('Test set: %d imgs, %d subjects\n',tot,tot_subj_test);

tot=0;
data={};
for i=tot_subj_test+1:numel(subjects_un)
    aux=find(subjects_files==subjects_un(i));
    for j=1:numel(aux)
        data{end+1}=files(aux(j)).name;
    end   
    tot=tot+numel(aux);
end
writecell(data,['data/',DATASET_NAME,'_learn.csv'],'Delimiter',',');
fprintf('Learn set: %d imgs, %d subjects\n',tot,numel(subjects_un)-tot_subj_test);


data_aux=data;

subjects_files = get_subjects_CASIA(data_aux);
subjects_un=unique(subjects_files);
subjects_un=subjects_un(randperm(numel(subjects_un)));

tot=0;
data={};
for i=1:round(numel(subjects_un)*PROPORTION_SUBJECTS_LEARN_GABOR)
    aux=find(subjects_files==subjects_un(i));
    for j=1:numel(aux)
        data{end+1}=data_aux{aux(j)};
    end   
    tot=tot+numel(aux);
end
writecell(data,['data/',DATASET_NAME,'_learn_Gabor.csv'],'Delimiter',',');
fprintf('Learn Gabor set: %d imgs, %d subjects\n',tot,round(numel(subjects_un)*PROPORTION_SUBJECTS_LEARN_GABOR));