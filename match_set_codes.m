function [scores_matrix, comps_matrix, AUC, dec]=match_set_codes(feats, files, subjects_files, TITLE, GENUINE_SMALLER_FLAG, FOLDER_RESULTS, FOLDER_NORMALIZED_IMGS)

scores_gen = [];
scores_imp = [];
comps_gen = {};
comps_imp = {};
tot_tests = 0;
valids=[];

FREQUENCY_SHOW=10000;

for i=1:size(feats,1)
    
    for j=i+1:size(feats,1)
        
        [score, tot_valid] = match_Gabor_features(feats(i,:), feats(j,:));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %             %   Keypoints-based      
        %
        %             idx_pairs = matchFeatures(feat_1,feat_2) ;
        %             %matched_1 = valid_1(idx_pairs(:,1));
        %             %matched_2 = valid_2(idx_pairs(:,2));
        %             if (subjects_files(i)==subjects_files(j))
        %                 scores_gen(end+1)=size(idx_pairs,1);
        %             else
        %                 scores_imp(end+1)=size(idx_pairs,1);
        %             end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if tot_valid<size(feats,2)*0.1
            continue
        end
        
        
        valids(end+1)=tot_valid;
        if (~isnan(score))
            if (subjects_files(i)==subjects_files(j))
                scores_gen(end+1) = score;
                comps_gen{end+1} = {files{i}(1:end-19), files{j}(1:end-19)};
                fprintf('>>>');
            else
                scores_imp(end+1) = score;
                comps_imp{end+1} = {files{i}(1:end-19), files{j}(1:end-19)};
            end
            
            fprintf('[%d]; %s (val %d)<->%s (val %d): %.2f\n',tot_tests+1, files{i}(1:end-19), sum(~isnan(feats(i,:))),files{j}(1:end-19), sum(~isnan(feats(j,:))), score);            
        end
        
        tot_tests = tot_tests+1;
        if ((mod(tot_tests,FREQUENCY_SHOW)==0) || ((i==numel(files)-1)&&(j==numel(files))))
            h=figure(70);
            clf;
            set(h,'Color','w');
            hold on;
            grid on;
            max_scale = max(max(scores_imp), max(scores_gen));
            XX=linspace(0, max_scale,50);
            TG=histc(scores_gen,XX);
            TG=TG./sum(TG);
            
            bar(XX,TG,0.5,'FaceColor',[0,1,0],'EdgeColor',[0,1,0]);
            TI=histc(scores_imp,XX);
            TI=TI./sum(TI);
            
            maxYY = max(max(TG), max(TI)) * 1.2;
            bar(XX,TI,0.25,'FaceColor',[1,0,0],'EdgeColor',[1,0,0]);
            axis([0 max_scale 0 maxYY]);
            dec = getDecidability([scores_gen', ones(numel(scores_gen),1); scores_imp', zeros(numel(scores_imp), 1)]);
            title(TITLE);
            legend(['dec=', sprintf('%.3f',dec),', tG=', num2str(numel(scores_gen)),', tI=',num2str(numel(scores_imp))]);
            %fprintf('Decidability %.3f\n', dec);
            
            
            h=figure(71);
            clf;
            set(h,'Color','w');
            [~, AUC, ~]=getROC2([scores_gen', ones(numel(scores_gen),1); scores_imp', zeros(numel(scores_imp), 1)], '-k',GENUINE_SMALLER_FLAG,1, 1, h);
            title(TITLE);
            legend(['AUC=', sprintf('%.4f',AUC)]);
            %fprintf('AUC %.4f\n ',AUC);
            
            h=figure(72);
            clf;
            set(h,'Color','w');
            hold on;
            grid on;
            XX=linspace(0, max(valids),50);
            TG=histc(valids,XX);
            bar(XX,TG,0.25,'FaceColor',[0,0,0],'EdgeColor',[0,0,0]);
            drawnow;
        end
    end
end

scores_matrix = [scores_gen', ones(numel(scores_gen),1); scores_imp', zeros(numel(scores_imp), 1)];
comps_matrix = [comps_gen'; comps_imp'];

if exist('FOLDER_RESULTS','var')

    tot_cases_plot = [100, 100, 100, 100];  %best_gen, worst_gen, best_imp, worst_imp
    save_notable_cases(scores_matrix, comps_matrix, FOLDER_RESULTS, FOLDER_NORMALIZED_IMGS, tot_cases_plot, 1);

    save([FOLDER_RESULTS, TITLE,'.mat'],'scores_matrix','comps_matrix');
end