

function [core,var,LC,L,btc,bt]=ARTDupdate(core,var,coreK,varK,num,ngmar,r,wk,L,LC,LK,LCK,lamda,Lapk,alpha)

    btc=min(wk,0.99*sqrt(LCK/LC));
    core=core+btc*(core-coreK);
    [V,LC]=gradcore(core,var,ngmar,r,num);
    core=PROXL1(V,lamda,1/(LC*r));
    
    for j=1:num
    bt(j)=min(wk,0.99*sqrt(LK(j)/L(j)));
    var{j}=var{j}+bt(j)*(var{j}-varK{j});
    [V,L(j)]=gradARTD(core,var,ngmar,r,j,num,Lapk,alpha(j));
    var{j}=PROXn1(V);
    %     var{j}=PROXL1(V,lamda,1/(L(j)*r));
    end

end


function [V,L]=gradARTD(core,var,ngmar,r,n,num,Lapk,alpha)
core=tensor(core);
index=1:num;
index(n)=[];
coreg=ttm(core,var,index);
tempB=double(tenmat(coreg,n));
temp=tempB*tempB';
Xn=double(tenmat(ngmar,n));
U=var{n}*temp-Xn*tempB'+alpha*Lapk{n}*var{n};
L=norm(temp,'fro')+alpha*norm(Lapk{n},'fro');
V=var{n}-1/(r*L)*U;
end

function x=PROXn1(x)
x=double(x);
x(x<0)=0;
end


