function write_codes_2_csv(path, codes, filenames)


f=fopen(path,'w');
for i=1:numel(filenames)
    if iscell(filenames)
        fprintf(f,'%s, ',filenames{i}(1:end-19));
    else
        fprintf(f,'%s, ',filenames(i).name(1:end-19));
    end
    for c=1:size(codes,2)
        if (isnan(codes(i, c)))
            fprintf(f,'9999');
        else
            fprintf(f,'%f',codes(i, c));
        end
        if (c<size(codes,2))
            fprintf(f,', ');
        end
    end
    fprintf(f,'\n');
end
fclose(f);