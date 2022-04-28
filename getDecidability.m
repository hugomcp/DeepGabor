function ret=getDecidability(scores)

idxG=find(scores(:,2)==1);
idxI=find(scores(:,2)==0);

ret=abs((mean(scores(idxG,1))-mean(scores(idxI,1))))/sqrt(std(scores(idxG,1)).^2+std(scores(idxI,1)).^2);