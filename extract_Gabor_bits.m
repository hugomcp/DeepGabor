function codes=extract_Gabor_bits(FOLDER_NORMALIZED_IMGS, files, kernels, positions, RESIZE_FACTOR, WIDTH_CIRCULAR_BAND, THRESHOLD_NOISE, BINARIZE)

codes=NaN(numel(files), numel(positions));

for i=1:numel(files)
    fprintf('Encoding %d/%d\n',i,numel(files));
        
    img=imread([FOLDER_NORMALIZED_IMGS, files{i}]);
    mask=imread([FOLDER_NORMALIZED_IMGS, files{i}(1:end-7),'mask.png']);
    img=img(:,:,1);
    mask=mask(:,:,1)./255;
    
    img=img(:, round(size(img,2)/2):end);       %use only the bottom part
    mask=mask(:, round(size(mask,2)/2):end);
    
    %img = [img, img(:, 1:WIDTH_CIRCULAR_BAND)]; %insert a circular band
    %mask = [mask, mask(:, 1:WIDTH_CIRCULAR_BAND)];
    
    codes(i, :) = extract_Gabor_features_img(img, mask, RESIZE_FACTOR, kernels, positions, THRESHOLD_NOISE, BINARIZE);
    
    
end



