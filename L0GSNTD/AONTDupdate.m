function [core,var,LC,L,btc,bt]=AONTDupdate(core,var,coreK,varK,num,ngmar,r,wk,L,LC,LK,LCK,Lapk,alpha,beta,lamda)

    btc=min(wk,0.99*sqrt(LCK/LC));
    core=core+btc*(core-coreK);
    [V,LC]=gradcore(core,var,ngmar,r,num);
    core=PROXn1(V);
    
    for j=1:num
    bt(j)=min(wk,0.99*sqrt(LK(j)/L(j)));
    var{j}=var{j}+bt(j)*(var{j}-varK{j});
    if(j<num)
    [V,L(j)]=gradAOTD(core,var,ngmar,r,j,num,Lapk,0,0,zeros(size(var{j},1),1));
    else
    [V,L(j)]=gradAOTD(core,var,ngmar,r,j,num,Lapk,alpha,beta,lamda);    
    end
    var{j}=PROXn1(V);
    end

end


function x=PROXn1(x)
x=double(x);
x(x<0)=0;
end

function [V,L]=gradAOTD(core,var,ngmar,r,n,num,Lapk,alpha,beta,lamda)
core=tensor(core);
index=1:num;
index(n)=[];
coreg=ttm(core,var,index);
tempB=double(tenmat(coreg,n));
temp=tempB*tempB';
Xn=double(tenmat(ngmar,n));



U=var{n}*temp-Xn*tempB'+alpha*var{n}*Lapk{n}+lamda*ones(size(var{n},2),1)';
U=U+beta*(var{n}*ones(size(var{n},2),1)-ones(size(var{n},1),1))*ones(size(var{n},2),1)';
L=norm(temp,'fro')+norm(alpha*Lapk{n}+beta*ones(size(var{n},2),1)*ones(size(var{n},2),1)','fro');
V=var{n}-1/(r*L)*U;
end


