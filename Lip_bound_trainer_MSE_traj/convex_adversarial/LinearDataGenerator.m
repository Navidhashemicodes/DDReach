clear   all
clc   
close   all

%%%  Neural network with 100 layers on a system with x \in R2  y \in R2  u
%%%  \in R1

%%%%%%%%%%%%%%%%%%%% The non-faulty plant that we dont know what is this,
%%%%%%%%%%%%%%%%%%%% and we need to learn it.
%%%%%%%%%%%%%%%%%  you should make it certain that the time step of
%%%%%%%%%%%%%%%%%  training data is exactly the same with the time step of
%%%%%%%%%%%%%%%%%  testing data, otherwise the learned model is
%%%%%%%%%%%%%%%%%  meaningless.

R2=0.0*[  0.0214    0.0112;0.0112    0.0217];



x1_min=0;
x1_max=10;
x1_discrete= 200;    %%% it captures [0 2*pi]
x1=linspace(x1_min,x1_max, x1_discrete);

x2_min=0;
x2_max=10;
x2_discrete=200; %%% it captures [0 10]
x2=linspace(x2_min,x2_max,x2_discrete);




v         = [1 1];
vLim      = [x1_discrete  x2_discrete];   
T1        =  x1_discrete*x2_discrete;
dim=2;

ready = false;
Nend=2;
N0=2;
j=0;
while ~ready
    j=j+1;
    eta=mvnrnd([0;0],R2,1)';
    Input(:,j) =[x1(v(1));x2(v(2))];
    y1=  0.5*x1(v(1))   +   0.3*x2(v(2))  +  eta(1,1);
    y2= -0.2*x2(v(2))   +   0.4*x1(v(1))  +  eta(2,1);
    Output(:,j)=[y1;y2];
    
   
    ready = true;
    ff=v(dim);
    for k = 1:dim                                   %index updater
        v(k) = v(k) + 1;
        if v(k) <= vLim(k)
            ready = false;
            break;
        end
        v(k) = 1;
    end
    
end
Xtrain=Input;
% Ytrain=Output; 
Ytrain=Output(1,:);    %%% I wanted to generate a 2 by 1 network to make it campatible with Dr Fazlyab
save('LinearData.mat','Xtrain','Ytrain')
