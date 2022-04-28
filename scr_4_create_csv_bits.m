clear;
clc;

DATASET_NAME='CASIA-Iris-Lamp';
SET_NAME='learn';

TYPE_OBJ='dec';

TOT_BITS=256;

cur_machine=pwd;
pos=find(cur_machine=='/');
cur_machine=cur_machine(pos(2)+1:pos(3)-1);
FOLDER_NORMALIZED_IMGS = ['/Users/',cur_machine,'/Desktop/',DATASET_NAME,'/segmented_normalized/'];

files=readcell(['data/',DATASET_NAME,'_',SET_NAME,'.csv']);

load(['data/ws_Gabor_configurations_',DATASET_NAME,'.mat']);    
load(['data/ws_selected_Gabor_configurations_',DATASET_NAME,'_',TYPE_OBJ,'.mat'],'selected_Gabor'); 

if (TOT_BITS<0)
    TOT_BITS=numel(selected_Gabor);
end

gabor_kernels=cell(1,TOT_BITS);
gabor_positions=zeros(1, TOT_BITS);
for i=1:TOT_BITS
    gabor_kernels{i}=gaborFilters{gaborConfigurations(selected_Gabor(i),1), gaborConfigurations(selected_Gabor(i),2)};
    gabor_positions(i)=gaborConfigurations(selected_Gabor(i),3);
end

existing_files=dir(['data/ws_Gabor_',DATASET_NAME,'_',SET_NAME,'_*_codes.mat']);
use=0;
for i=1:numel(existing_files)
    [tot_bits_tmp]=strread(existing_files(i).name, ['ws_Gabor_',DATASET_NAME,'_',SET_NAME,'_%d_codes.mat']);
    if (tot_bits_tmp>=TOT_BITS)
        use=1;
        break;
    end
end

if (use==0)
    codes=extract_Gabor_bits(FOLDER_NORMALIZED_IMGS, files, gabor_kernels, gabor_positions, RESIZE_FACTOR, WIDTH_CIRCULAR_BAND, THRESHOLD_NOISE, false);        
else    
    load(['data/ws_Gabor_',DATASET_NAME,'_',SET_NAME,'_',num2str(tot_bits_tmp),'_codes.mat']);     
    codes=codes(:, 1: TOT_BITS);
end

subjects_files=get_subjects_CASIA(files);
subjects=unique(subjects_files);

[gaborCodes, tot_ok, tot_ko, tot_invalidated]=correct_bits(codes, subjects_files, subjects);

fprintf('Valid %.2f%%. Corrected %.2f%% bits...',(tot_ok+tot_ko)/numel(codes)*100, tot_ko/(tot_ok+tot_ko)*100);
save(['data/ws_Gabor_',DATASET_NAME,'_',SET_NAME,'_',num2str(TOT_BITS),'_codes.mat'],'codes','subjects','subjects_files');

%return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% write the "csv" files

TOT_BITS=256;
write_codes_2_csv(['data/features_Gabor_',DATASET_NAME,'_',SET_NAME,'_in_',num2str(TOT_BITS),'_bits.csv'], gaborCodes(:,1:TOT_BITS,1), files);
write_codes_2_csv(['data/features_Gabor_corrected_',DATASET_NAME,'_',SET_NAME,'_in_',num2str(TOT_BITS),'_bits.csv'], gaborCodes(:,1:TOT_BITS,2), files);

kind_bits=-(2*(gaborCodes(:, 1:TOT_BITS, 1)==gaborCodes(:, 1:TOT_BITS, 2))-1);
kind_bits(isnan(gaborCodes(:,1:TOT_BITS,1)))=NaN;
write_codes_2_csv(['data/features_Gabor_kind_bits_',DATASET_NAME,'_',SET_NAME,'_in_',num2str(TOT_BITS),'_bits.csv'], kind_bits, files);

write_codes_2_csv(['data/features_Gabor_majority_',DATASET_NAME,'_',SET_NAME,'_in_',num2str(TOT_BITS),'_bits.csv'], gaborCodes(:,1:TOT_BITS,3), files);

regions=[];
for i=1:TOT_BITS
    pos=gaborConfigurations(selected_Gabor(i),3);
    kernel=size(gaborFilters{gaborConfigurations(selected_Gabor(i),1),1},1);    
    ret=position_2_input_region(pos, kernel, [64, 256], RESIZE_FACTOR);
    regions(end+1,:)=ret;
end
csvwrite(['data/inf_filters_',DATASET_NAME,'_',num2str(TOT_BITS),'.csv'],regions);

regions=[];
for i=1:TOT_BITS
    regions(end+1,:)=[gaborConfigurations(selected_Gabor(i),1), gaborConfigurations(selected_Gabor(i),2)];
end
csvwrite(['data/idx_filters_',DATASET_NAME,'_',num2str(TOT_BITS),'.csv'],regions);


