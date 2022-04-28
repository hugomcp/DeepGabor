function write_cell_2_csv(path, cel)

f=fopen(path,'w');
for i=1:size(cel,1)    
    fprintf(f,'%s, ',cel{i,1});    
    for c=2:size(cel,2)
        fprintf(f,'%f',cel{i, c});
        if (c<size(cel,2))
            fprintf(f,', ');
        end
    end
    fprintf(f,'\n');
end
fclose(f);