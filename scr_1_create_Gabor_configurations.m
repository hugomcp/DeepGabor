%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 22-12-2021.
% Create all possible Gabor features, by combining all kernels at every
% position of the normalized images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

DATASET_NAME='CASIA-Iris-Lamp';

cur_machine=pwd;
pos=find(cur_machine=='/');
cur_machine=cur_machine(pos(2)+1:pos(3)-1);
FOLDER_GABOR = ['/Users/',cur_machine,'/Desktop/',DATASET_NAME,'/Gabor/'];

FOLDER_NORMALIZED_IMGS = ['/Users/',cur_machine,'/Desktop/',DATASET_NAME,'/segmented_normalized/'];

WAVELENGTH_GABOR = [4,4*sqrt(2),8,8*sqrt(2),16];
ORIENTATION_GABOR = [0, pi/4, pi/2, 3*pi/4];
PHASE_GABOR = [0,pi/2];
RATIO_GABOR = [1];
RESIZE_FACTOR = 0.1;
THRESHOLD_NOISE = 0.1;
gaborFilters=get_Gabor_filters(WAVELENGTH_GABOR,ORIENTATION_GABOR,PHASE_GABOR,RATIO_GABOR);

WIDTH_CIRCULAR_BAND = 0;
for i=1:size(gaborFilters,1)
    WIDTH_CIRCULAR_BAND = max(WIDTH_CIRCULAR_BAND, size(gaborFilters{i,1},1));
end

%% PHASE 1: Encode the images with all possible features

if (~exist(FOLDER_GABOR,'dir'))
    mkdir(FOLDER_GABOR);
end

files = readcell(['data/',DATASET_NAME,'_learn_Gabor.csv']);


SIZE_OUT_IMGS=0;
results=cell(size(gaborFilters,1),2);

RESIZE_FACTOR=1;
for i=1:numel(files)
    fprintf('Encoding %d/%d: %s\n',i,numel(files), files{i});
    
%     if (exist([FOLDER_GABOR,files{i}(1:end-3),'mat'],'file'))
%         continue;
%     end
    
    img=imread([FOLDER_NORMALIZED_IMGS, files{i}]);
    mask=imread([FOLDER_NORMALIZED_IMGS, files{i}(1:end-7),'mask.png']);
    img=img(:,:,1);
    mask=mask(:,:,1)./255;
    
    img=img(:, round(size(img,2)/2):end);       %use only the bottom part
    mask=mask(:, round(size(mask,2)/2):end);
    mask(:)=1;
    
    %img = [img, img(:, 1:WIDTH_CIRCULAR_BAND)];  %insert a circular band
    %mask = [mask, mask(:, 1:WIDTH_CIRCULAR_BAND)];
    
    results = extract_Gabor_configurations(img, mask, gaborFilters, RESIZE_FACTOR, THRESHOLD_NOISE);
    
    save([FOLDER_GABOR,files{i}(1:end-3),'mat'],'results');
end

% Create all combinations of filters/positions

files = readcell(['data/',DATASET_NAME,'_learn_gabor.csv']);

gaborConfigurations = [];
load([FOLDER_GABOR,files{1}(1:end-3),'mat']);
for i=1:size(gaborFilters,1)
    pos=1:numel(results{i,1});
    gaborConfigurations=[gaborConfigurations; repmat(i,numel(pos),1), repmat(1,numel(pos),1), pos'];
    gaborConfigurations=[gaborConfigurations; repmat(i,numel(pos),1), repmat(2,numel(pos),1), pos'];
end

save(['data/ws_Gabor_configurations_',DATASET_NAME,'.mat'],'gaborFilters','WIDTH_CIRCULAR_BAND','gaborConfigurations','RESIZE_FACTOR','THRESHOLD_NOISE');
