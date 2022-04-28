function [ret, totv]=match_Gabor_features(feat_1, feat_2)

valid = find(~isnan(feat_1)&~isnan(feat_2));

if (numel(valid)==0)
    ret = NaN;
else
    ret = sum(feat_1(valid)~=feat_2(valid))/numel(valid);
end

totv=numel(valid);