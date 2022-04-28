function ret=extract_Gabor_features_img(img, mask, RESIZE_FACTOR, kernels, positions, threshold_noise, binarize)



ret = zeros(numel(kernels),1);
for j=1:numel(kernels)
    feat = imresize(conv2(img, kernels{j},'valid'), RESIZE_FACTOR);
    mask_out = imresize(conv2(1-mask, ones(size(kernels{j},1))./numel(kernels{j}),'valid'), RESIZE_FACTOR);
    
    feat(mask_out > threshold_noise) = nan;
    
    ret(j)=feat(positions(j));
    
    %     h=figure(1);
    %     clf;
    %     imshow(feat,[],'InitialMagnification',500);
    %     hold on;
    %     [r,c]=ind2sub(size(feat),configs(j,3));
    %     plot(c,r,'xr','LineWidth',5,'MarkerSize',10);
    
end

if (binarize)
    ret(ret>0)=1;
    ret(ret<=0)=0;
end


