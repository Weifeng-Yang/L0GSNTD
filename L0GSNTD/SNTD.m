%%  All parameters of this function are explained the same as 'main_Run_me' and 'ALGOchoose' functions
function [core,var,loss,timerun]=SNTD(core,var,ngmar,maxiteropt,stopindex,r,lamda)
%% initialization algorithm
loss=[];
timerun=[0];
num=length(size(ngmar));
LK=zeros(1,num);
LCK=0;
rate=0;
L=ones(1,num);
LC=1;
tk=1;

varK=var;
coreK=core;
wk=(tk-1)/(tk);

returnloss=norm(tensor(ngmar));
loss(1)=computeloss(ngmar,core,var,lamda);


t1=clock;


for i=1:maxiteropt
%% update parameters
fprintf("%d\n",i);
bt=zeros(1,num);
btc=0;

for j=1:num
LtempK=LK;
LCtempK=LCK;
vartempK=varK;
coretempK=coreK;    
varK=var;
coreK=core;
LK=L;
LCK=LC;    
[core,var,LC,L,btc,bt]=SNTDupdate(core,var,coretempK,vartempK,num,ngmar,r,wk,L,LC,LtempK,LCtempK,lamda,j);
loss(i+1)=computeloss(ngmar,core,var,lamda);


%% Judging whether to extrapolate
if(loss(i+1)>=loss(i))
    var=varK;
    core=coreK;
    L=LK;
    LC=LCK;
    [core,var,LC,L]=SNTDupdate(core,var,coretempK,vartempK,num,ngmar,r,0,L,LC,LtempK,LCtempK,lamda,j);
    loss(i+1)=computeloss(ngmar,core,var,lamda);
end
end


%% Check if termination condition is met
fprintf("SNTD\n");
check1=norm(tensor(core));
check2=norm(tensor(coreK));
for j=1:num
    fprintf("nonzero:%d\n",nnz(var{j}));  
    check1=check1+norm(var{j}-varK{j},'fro');
    check2=check2+norm(varK{j},'fro');
end

bts{i}=[btc,bt];
t2=clock;
timerun(i+1)=etime(t2,t1);
% Res=abs(loss(i+1)-loss(i));
Res=check1/check2;
fprintf("cri：%d\n",Res);
stop=stopcheck(Res,timerun,stopindex);
if(stop==1)
    fprintf("Number of terminations：%d\n",i);
    pause(4);
    break;
end


tk=(1+sqrt(1+4*tk^2))/2;
wk=(tk-1)/(tk);

end


end





function loss=computeloss(ngmar,core,var,lamda)
    loss=compute(core,var,ngmar)+lamda/2*norm(tensor(core))^2;
    for i=1:length(size(ngmar))-1
    loss=loss+lamda/2*norm(var{i},'fro')^2;
    end
end


