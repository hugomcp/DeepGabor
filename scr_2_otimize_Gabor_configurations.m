%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 13-10-2021.
% Selects the optimal Gabor filter configurations, using a SFS approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

DATASET_NAME='CASIA-Iris-Thousand';

SET_NAME='learn_Gabor';

cur_machine=pwd;
pos=find(cur_machine=='/');
cur_machine=cur_machine(pos(2)+1:pos(3)-1);
FOLDER_OUT = ['/Users/',cur_machine,'/Desktop/',DATASET_NAME,'/Gabor/'];

MIN_SUPPORT_VALID_BIT = 0.75;

TYPE_OBJ='dec';


%% PHASE 2: Select the optimal filters configuration

files=readcell(['data/',DATASET_NAME,'_',SET_NAME,'.csv']);
subjects_files  = get_subjects_CASIA(files);
all_signatures=cell(numel(files),1);
for i=1:numel(files)
    load([FOLDER_OUT, files{i}(1:end-3)],'results');
    all_signatures{i} = results;
end


if (exist(['data/ws_selected_Gabor_configurations_',DATASET_NAME,'_',TYPE_OBJ,'.mat'],'file'))
    load(['data/ws_selected_Gabor_configurations_',DATASET_NAME,'_',TYPE_OBJ,'.mat'],'selected_Gabor','obj');    
else
    selected_Gabor=[];
    obj=[];
end


type_comps=(pdist(subjects_files)==0)';

load(['data/ws_Gabor_configurations_',DATASET_NAME,'.mat'],'gaborConfigurations');
dists_all_individual=zeros(numel(type_comps),size(gaborConfigurations,1));
v=zeros(numel(all_signatures),1);

elegible=ones(numel(gaborConfigurations),1);
for b=1:size(gaborConfigurations,1)
    fprintf('Collecting initial data %d/%d\n',b, size(gaborConfigurations,1));
    for i=1:numel(all_signatures)
        v(i)=all_signatures{i}{gaborConfigurations(b,1), gaborConfigurations(b,2)}(gaborConfigurations(b,3));        
    end
    v(v>0)=1;
    v(v<=0)=0;
    dists_all_individual(:,b)=pdist(v, 'hamming');
    valid_idx=~isnan(dists_all_individual(:,b));
    if (sum(type_comps(valid_idx)==1)<sum(type_comps==1)*MIN_SUPPORT_VALID_BIT)||(sum(type_comps(valid_idx)==0)<sum(type_comps==0)*MIN_SUPPORT_VALID_BIT)
        elegible(b)=0;
    end
end

dists_acc=zeros(size(type_comps,1),2);
if (numel(selected_Gabor)>0)
    dists_acc(:,1)=nansum(dists_all_individual(:, selected_Gabor),2);
    dists_acc(:,2)=sum(~isnan(dists_all_individual(:, selected_Gabor)),2);
end


while(numel(selected_Gabor)< size(gaborConfigurations,1))
    
    merit = zeros(size(gaborConfigurations,1),1);
    WaitMessage = parfor_wait(size(gaborConfigurations,1));
    
    
    valid_prev=(dists_acc(:,2)>0);
            
    tot_prev=dists_acc(:,2);
    %poolObj=parpool('local', 8);
    dist1=dists_acc(:,1);
    for j=1:size(gaborConfigurations,1)
        %fprintf('Try %d\n',j);
        if (~elegible(j))
            continue;
        end
        if (sum(selected_Gabor==j)>0)
            continue;
        end
        
        dists=dist1;
        
        val_cur=~isnan(dists_all_individual(:, j));               
        dists(val_cur)=(dists(val_cur)+dists_all_individual(val_cur, j))./(tot_prev(val_cur)+1);                
        dists(~val_cur)=dists(~val_cur)./tot_prev(~val_cur);
        
        if (strcmp(TYPE_OBJ,'roc')==1)        
            [~ , ~, ~, ret]=perfcurve(type_comps(valid_prev|val_cur), dists(valid_prev|val_cur),0);
        else
            ret=(mean(dists((valid_prev|val_cur)&(type_comps==0)))-mean(dists((valid_prev|val_cur)&(type_comps==1))))./...
                sqrt(std(dists((valid_prev|val_cur)&(type_comps==0))).^2+std(dists((valid_prev|val_cur)&(type_comps==1))).^2);
        end
    
        if (isinf(ret))
            ret=0;
        end
        
        merit(j)=ret;
        
        WaitMessage.Send;
        pause(0.002);
    end
    WaitMessage.Destroy
    %delete(poolObj);
        
    [best, best_idx] = max(merit);
                
    val_cur=~isnan(dists_all_individual(:, best_idx));
    dists_acc(val_cur,1)=dists_acc(val_cur,1)+dists_all_individual(val_cur, best_idx); 
    dists_acc(val_cur,2)=dists_acc(val_cur,2)+1;
    
    selected_Gabor(end+1) = best_idx;
    
    fprintf('Iteration %d. Selected configurations: [',numel(selected_Gabor));
    for i=1:numel(selected_Gabor)
        fprintf('%d ',selected_Gabor(i));
    end
    fprintf(']. OBJ=%.2f\n', best);
    obj=[obj, best];
    
    save(['data/ws_selected_Gabor_configurations_',DATASET_NAME,'_',TYPE_OBJ,'.mat'],'selected_Gabor','obj');
end




