
% Input.
% var,core    : initial matrix and core tensor
% ngmar       : decomposed tensor
% coreaa,aa   : Maximum number of non-zero elements for each decomposition matrix and core tensor
% The remaining parameters are explained the same as the 'main_Run_me' function

% Output.
% cores, vars : Decomposition matrix and core tensor resulting from the final iterative result
% loss:       : Array of loss functions generated during iteration
% tr:         : Runtime array during iteration

function [data,varss]=ALGOchoose(core,var,ngmar,coreaa,aa,maxiteropt,Rdims,flag,stopindex,r,alphat)

if(flag==1)
[vars,loss,tr]=L0NMF(var,ngmar,aa,maxiteropt,stopindex,r);
varss{1}=vars;
lossdata=loss;
trdata=tr;
 
  

elseif(flag==2)
[vars,loss,tr]=L1GSNMF(var,ngmar,maxiteropt,stopindex,r,1);
varss{1}=vars;
lossdata=loss;
trdata=tr;
 
  





elseif(flag==3)
[cores,vars,loss,tr]=SNTD(core,var,ngmar,maxiteropt,stopindex,r,0.5);
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  


elseif(flag==4) 
     
[cores,vars,loss,tr]=ARTD(core,var,ngmar,maxiteropt,stopindex,r,0);
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  

elseif(flag==5) 
    
[cores,vars,loss,tr]=ARTD(core,var,ngmar,maxiteropt,stopindex,r,0.5);
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  



elseif(flag==6) 
[cores,vars,loss,tr]=AONTD(core,var,ngmar,maxiteropt,stopindex,r,1000,1000);
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  



elseif(flag==7) 
    
[cores,vars,loss,tr]=GSNTD(core,var,ngmar,maxiteropt,Rdims,stopindex,r,1.7,10);
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  

elseif(flag==8) 
    
[cores,vars,loss,tr]=L1GSNTD(core,var,ngmar,maxiteropt,stopindex,r,0.5,0.99);
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  


elseif(flag==9) 
[cores,vars,loss,tr]=L0GSNTD(core,var,ngmar,coreaa,aa,maxiteropt,Rdims,stopindex,r,0,alphat,0.99);  
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  
   


elseif(flag==10) 
[cores,vars,loss,tr]=L0GSNTD(core,var,ngmar,coreaa,aa,maxiteropt,Rdims,stopindex,r,1,alphat,0.99);  
varss{1}=cores;
varss{2}=vars;
lossdata=loss;
trdata=tr;
 
  
end


data{1}=lossdata;
data{2}=trdata;






end



