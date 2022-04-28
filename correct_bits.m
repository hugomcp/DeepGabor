function [returned, tot_ok, tot_ko, tot_invalidated]=correct_bits(codes, subjects_files, subjects)

tot_ok=0;
tot_ko=0;
tot_invalidated=0;
returned=NaN(size(codes,1), size(codes,2),3);
returned(:,:,1)=codes;
for b=1:size(codes,2)
    fprintf('Bit %d/%d\n',b,size(codes,2));
    for s=1:numel(subjects)
        
        idx = find(subjects_files==subjects(s));
        obs=returned(idx,b,1);
        
        tot_pos=sum(obs>0);
        tot_neg=sum(obs<=0);
        
        if tot_pos>tot_neg            
            to_correct = find(obs<=0);
            ok = find(obs>0);
            returned(idx(to_correct), b, 2) = median(obs(ok));
            returned(idx(ok), b, 2) = obs(ok);
            tot_ok=tot_ok+tot_pos;
            tot_ko=tot_ko+tot_neg;
            returned(idx, b, 3)=sign(median(obs(ok)));
        else
            if tot_neg>tot_pos
                to_correct = find(obs>0);
                ok = find(obs<=0);
                returned(idx(to_correct), b, 2) = median(obs(ok));
                returned(idx(ok), b, 2) = obs(ok);
                tot_ok=tot_ok+tot_neg;
                tot_ko=tot_ko+tot_pos;
                returned(idx, b, 3)=sign(median(obs(ok)));
            else
                tot_invalidated=tot_invalidated+sum(~isnan(returned(idx,b,1)));
                returned(idx, b, 2)=NaN;                
            end
        end                
    end
end
