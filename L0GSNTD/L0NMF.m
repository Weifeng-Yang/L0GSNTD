%%  All parameters of this function are explained the same as 'main_Run_me' and 'ALGOchoose' functions
function [var,loss,timerun]=L0NMF(var,ngmar,aa,maxiteropt,stopindex,r)
%% initialization algorithm
loss=[];
rate=0;
timerun=[0];
ngmar=double(tenmat(ngmar,length(size(ngmar))))';
num=length(size(ngmar));
R=size(var{1},2);
percent=aa(1)/prod(size(var{1}));
aa=[];
var=[];

for i=1:num
    var{i}=rand(size(ngmar,i),R);
    aa(i)=round(size(ngmar,i)*R*percent);
end
var{num}=var{num}';

for i=1:num
    varze{i}=zeros(size(var{i},1),size(var{i},2));
end
LK=zeros(1,num);
L=ones(1,num);
tk=1;
bts=[];
varK=var;

loss(1)=compute(var,num,ngmar);
returnloss=norm(ngmar,"fro");
wk=zeros(1,num);
for j=1:num
wk(j)=(tk-1)/(tk);
end
t1=clock;


for i=1:maxiteropt
%% update parameters
fprintf("%d\n",i);
vv=var;
for j=1:num
    if(j==1)
    wk(j)=min(wk(j),0.9999*(r-1)/(2*r+2)*sqrt(LK(j)/L(j)));
    vv{j}=var{j}+wk(j)*(var{j}-varK{j});
    varK{j}=var{j};
    LK(j)=L(j);
    [V,L(j)]=grad(vv,ngmar,j,r);
    var{j}=PROX(varze{j},V,aa(j));
    else
    wk(j)=min(wk(j),0.9999*sqrt(LK(j)/L(j)));
    vv{j}=var{j}+wk(j)*(var{j}-varK{j});
    varK{j}=var{j};
    LK(j)=L(j);
    [V,L(j)]=grad(vv,ngmar,j,r);
    var{j}=PROXn1(V);    
    end

    vv{j}=var{j};
end
loss(i+1)=compute(var,num,ngmar);

%% Judging whether to extrapolate
if(loss(i+1)>loss(i))
    var=varK;
    for j=1:num
    [V,L(j)]=grad(var,ngmar,j,r);
    if(j==1)
    var{j}=PROX(varze{j},V,aa(j));
    else
    var{j}=PROXn1(V);    
    end
    end
    loss(i+1)=compute(var,num,ngmar);
end



%% Check if termination condition is met
fprintf("L0NMF\n");
check1=0;
check2=0;
for j=1:num
    fprintf("nonzero:%d\n",nnz(var{j}));
    check1=check1+norm(var{j}-varK{j},'fro');
    check2=check2+norm(varK{j},'fro');
end
bts{i}=wk;
t2=clock;
timerun(i+1)=etime(t2,t1);
% Res=abs(loss(i+1)-loss(i));
% Res=norm(tensor(S2-S1))/returnloss;
% S1=S2;
Res=check1/check2;
% Res=norm(tensor(S2-S1))/returnloss;
fprintf("Rel：%d\n",Res);
stop=stopcheck(Res,timerun,stopindex);
if(stop==1)
    fprintf("Number of terminations：%d\n",i);
    pause(2);
    break;
end


tk=(1+sqrt(1+4*tk^2))/2;
for j=1:num
wk(j)=(tk-1)/(tk);
end


end

end











function x=PROXn1(x)
x=double(x);
x(x<0)=0;
end


function loss=compute(var,num,ngmar)
    nga=var{1};
    for i=2:num
        nga=nga*var{i};
    end
    loss=norm(ngmar-nga,'fro');
end


function [U,L]=grad(var,ngmar,n,r)
    if(n==1)
    ZT=var{n+1};
    ck=norm(ZT*ZT','fro');
    ck=checkck(ck);
    mar=var{n}*ZT;
    U=var{n}-1/(r*ck)*(mar-ngmar)*ZT'; 
    else
    Z=var{1};
    ck=norm(Z'*Z,'fro');
    ck=checkck(ck);
    mar=Z*var{n};
    U=var{n}-1/(r*ck)*Z'*(mar-ngmar);   
    end
    L=ck;
end


function ck=checkck(ck)
    if(ck==0) 
        a=0;
       while(a==0)
           a=rand(1);
       end
       ck=a;
     end
end

