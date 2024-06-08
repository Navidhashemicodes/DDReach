function [theInput, theOutput, maxmin] = vdp_nln_Datagenerator_ss(lb, ub, timestep, normalization, num_traj , horizon, avg_m, cov_m, avg_s, cov_s) 

n=2;


Nend=(horizon+1)*n;
N0=n;
Input = zeros(N0, num_traj);
Output = zeros(Nend, num_traj);

parfor j = 1:num_traj
    F = zeros(Nend,1);
    initial=lb + rand(n,1).*(ub-lb);
    nu = mvnrnd(avg_m , cov_m , 1 )';
    Input(:,j) = initial+nu;
    F(1:n,1) = initial+nu;
    for i=1:horizon
        noise = mvnrnd(avg_s, cov_s, 1);
        [~,in_out] =  ode45(@(t,x)vannderpol(t,x, noise),[0 timestep],initial');
        in_out=in_out(end,:)';
        index = n*i;
        nu = mvnrnd(avg_m , cov_m , 1 )';
        F(index+1:index+n,1) = in_out+nu;
        initial = in_out;
    end
    Output(:,j) = F;
    
end

if normalization==1
    
    a = -1;
    b =  1;
    maxin = max(Input,[],2);
    maxmin.maxin=maxin;
    minin = min(Input,[],2);
    maxmin.minin=minin;
    maxout= max(Output,[],2);
    minout= min(Output,[],2);
    theInput = (b-a) * diag(1./ (maxin-minin) ) * ( Input - minin )  + a ;
    theOutput= (b-a) * diag(1./(maxout-minout)) * (Output - minout)  + a ;
elseif normalization==0
    theInput=Input;
    theOutput=Output;
    maxmin='no normalization';
end
end