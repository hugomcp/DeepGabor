%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 18-10-2021.
% Script to evaluate a recognition method and show the results on-the-fly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear;
clc;

DATASET_NAME_TEST='CASIA-Iris-Lamp';
SET_NAME='test';

DATASET_NAME_FEATURES='CASIA-Iris-Lamp';
TYPE_OBJ='dec';

cur_machine=pwd;
pos=find(cur_machine=='/');
cur_machine=cur_machine(pos(2)+1:pos(3)-1);

FOLDER_NORMALIZED_IMGS = ['/Users/',cur_machine,'/Desktop/',DATASET_NAME_TEST,'/segmented_normalized/'];

TITLE = 'Daugman';

FOLDER_RESULTS =['results/',TITLE,'/'];

GENUINE_SMALLER_FLAG = 1;

TOT_BITS=256;

if (~exist(FOLDER_RESULTS,'dir'))
    mkdir(FOLDER_RESULTS);
end

files=readcell(['data/',DATASET_NAME_TEST,'_',SET_NAME,'.csv']);
subjects_files=get_subjects_CASIA(files);
subjects=unique(subjects_files);

load(['data/ws_gabor_Configurations_',DATASET_NAME_FEATURES,'.mat']);
load(['data/ws_selected_Gabor_configurations_',DATASET_NAME_FEATURES,'_',TYPE_OBJ,'.mat'],'selected_Gabor');  

if (TOT_BITS<0)
    TOT_BITS=numel(selected_Gabor);
end

gabor_kernels=cell(1,TOT_BITS);
gabor_positions=zeros(1, TOT_BITS);
for i=1:TOT_BITS
    gabor_kernels{i}=gaborFilters{gaborConfigurations(selected_Gabor(i),1), gaborConfigurations(selected_Gabor(i),2)};
    gabor_positions(i)=gaborConfigurations(selected_Gabor(i),3);
end


if (~exist(['data/ws_Gabor_',DATASET_NAME_TEST,'_',SET_NAME,'_2048_codes.mat'],'file'))            
    codes=extract_Gabor_bits(FOLDER_NORMALIZED_IMGS, files, gabor_kernels, gabor_positions, RESIZE_FACTOR, WIDTH_CIRCULAR_BAND, THRESHOLD_NOISE, true);
else
    load(['data/ws_Gabor_',DATASET_NAME_TEST,'_',SET_NAME,'_2048_codes.mat'],'codes');     
end

codes(codes>0)=1;
codes(codes<0)=0;
codes=codes(:, 1:TOT_BITS);

[scores_matrix, comps_matrix, AUC, dec]=match_set_codes(codes, files, subjects_files, TITLE, GENUINE_SMALLER_FLAG, FOLDER_RESULTS, FOLDER_NORMALIZED_IMGS);




