function results=extract_Gabor_configurations(img, mask, gaborFilters, RESIZE_FACTOR, THRESHOLD_NOISE)

results=cell(size(gaborFilters,1),2);
for j=1:size(gaborFilters,1)
    realPart=conv2(img, gaborFilters{j,1},'valid');
    imagPart=conv2(img, gaborFilters{j,2},'valid');
    mask_out = conv2(1-mask, ones(size(gaborFilters{j,1},1))./numel(gaborFilters{j,1}),'valid');
    
    mask_out = imresize(mask_out, RESIZE_FACTOR);
    imagPart = imresize(imagPart, RESIZE_FACTOR);
    realPart = imresize(realPart, RESIZE_FACTOR);
        
    
    realPart(mask_out > THRESHOLD_NOISE) = nan;
    imagPart(mask_out > THRESHOLD_NOISE) = nan;
    
    results{j,1}=realPart;
    results{j,2}=imagPart;
    
end