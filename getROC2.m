function [ROC, AUC, decidability]=getROC2(scores,style,sign,plt, eer, handleFig)
%plt: 0 No, 1=normal scale, 2=logx


thresholds=sort(unique(scores(:,1)));

if (numel(thresholds>998))
    thresholds=thresholds(round(linspace(1,numel(thresholds),998)));   
end

thresholds=[min(thresholds)-0.1; thresholds; max(thresholds)+0.1];



ROC=zeros(numel(thresholds)-1,3);

for i=1:size(ROC,1)
    ROC(i,1)=(thresholds(i+1)+thresholds(i))/2;
    if (sign==1)
        ROC(i,2)=sum((scores(:,2)==1)&(scores(:,1)<=ROC(i,1)))/sum(scores(:,2)==1);
        ROC(i,3)=sum((scores(:,2)==0)&(scores(:,1)<=ROC(i,1)))/sum(scores(:,2)==0);
    else
        ROC(i,2)=sum((scores(:,2)==1)&(scores(:,1)>=ROC(i,1)))/sum(scores(:,2)==1);
        ROC(i,3)=sum((scores(:,2)==0)&(scores(:,1)>=ROC(i,1)))/sum(scores(:,2)==0);
    end
end


if (plt~=0)
    %figure('Color','w');
    figure(handleFig);
    if (plt==2)
        semilogx(ROC(:,3),ROC(:,2),style,'LineWidth',1);
    else
        plot(ROC(:,3),ROC(:,2),style,'LineWidth',1);
    end
    hold on, grid on;
    %    ylabel('True Positive','FontSize',14);
    %    xlabel('False Positive','FontSize',14);
    if (eer==1)
        if (plt==2)
            semilogx(min(ROC(:,3)):1e-3:1,min(ROC(:,3)):1e-3:1,'-.k','LineWidth',2);
        else
            plot(min(ROC(:,3)):1e-3:1,min(ROC(:,3)):1e-3:1,'-.k','LineWidth',2);
        end
    end
    axis([0.0 1 0 1]);
end

AUC=0;
format long;
for i=2:size(ROC,1)
    AUC=AUC+abs(ROC(i,3)-ROC(i-1,3))*min(ROC(i,2),ROC(i-1,2))+...
        abs(ROC(i,3)-ROC(i-1,3))*(max(ROC(i,2),ROC(i-1,2))-min(ROC(i,2),ROC(i-1,2)))/2;
    
%        disp([num2str(i),' Parcela ',num2str(abs(ROC(i,3)-ROC(i-1,3))*min(ROC(i,2),ROC(i-1,2))+...
%            abs(ROC(i,3)-ROC(i-1,3))*(max(ROC(i,2),ROC(i-1,2))-min(ROC(i,2),ROC(i-1,2)))/2),' = ',num2str(AUC)]);
    
    
end


decidability=abs(mean(scores(scores(:,2)==1,1))-mean(scores(scores(:,2)==0,1)))/...
    sqrt(0.5*(std(scores(scores(:,2)==1,1))^2+std(scores(scores(:,2)==0,1))^2));

%disp(AUC);
%disp(decidability);


Col3=unique(ROC(:,3));
ROC_temp=zeros(numel(Col3),3);
for i=1:numel(Col3)
    ROC_temp(i,3)=Col3(end-i+1);
    ROC_temp(i,2)=max(ROC(ROC(:,3)==Col3(end-i+1),2));
    ROC_temp(i,1)=max(ROC(ROC(:,3)==Col3(end-i+1),1));
end
ROC=ROC_temp;


%legend(['AUC: ',num2str(AUC),' decidab.: ',num2str(decidability)],4);

%legend('Fusion','HOG','LBP','SIFT','Gabor',4)


