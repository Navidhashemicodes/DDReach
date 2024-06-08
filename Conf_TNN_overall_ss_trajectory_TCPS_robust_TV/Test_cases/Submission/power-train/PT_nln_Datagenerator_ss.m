function [theInput, theOutput, maxmin] = PT_nln_Datagenerator_ss(lb, ub, timestep, normalization, num_traj , simTime, avg_m, cov_m)

n=5;

horizon = simTime/timestep;

Nend=(horizon+1)*n;
N0=n;
Input = zeros(N0, num_traj);
Output = zeros(Nend, num_traj);


c1=0.41328; %#ok<*NASGU>
c2=-0.366;
c3=0.08979;
c4=-0.0337;
c5=0.0001;
c6=2.821;
c7=-0.05231;
c8=0.10299;
c9=-0.00063;
c10=1.0;
c11=14.7;
c12=0.9;
c13=0.04;
c14=0.14;
% total simulation time
% time to start measurement, mainly used to ignore
% Simulink initialization phase
measureTime = 1;
fault_time = 100;
en_speed = 1000;


c23=1.0;
c24=1.0;
c25=1.0;

t = 0:timestep:simTime;
t_len = length(t);


for j = 1:num_traj
    

    
    
    initial=lb + rand(n,1).*(ub-lb);

    p_init      = initial(1);
    lambda_init = initial(2);
    p_est_init  = initial(3);
    i_init      = initial(4); 
    mode_init   = initial(5);



    

    sim("AbstractFuelControl_M2_Navid.slx")
    % % pick time horizon of 20, only track reach states after time 10
    % % 0.02, 0.05, 0.1 (objective: reduce conservatism)

    T = ScopeData.time;
    T_len = length(T);
    S = zeros(t_len,5);

    S(:,1) = interp1(T , ScopeData.signals(2).values , t');
    S(:,2) = interp1(T,  ScopeData.signals(3).values , t');
    S(:,3) = interp1(T,  ScopeData.signals(4).values , t');
    S(:,4) = interp1(T,  ScopeData.signals(5).values , t');
    S(:,5) = interp1(T,  ScopeData.signals(6).values , t');


    F = reshape(S', [5*t_len , 1]);

    nu = mvnrnd(avg_m , cov_m , 1 )';
    Input(:,j) = initial+nu;
    F(1:n,1) = F(1:n,1) + nu;
    for i=1:horizon
        index = n*i;
        nu = mvnrnd(avg_m , cov_m , 1 )';
        F(index+1:index+n,1) = F(index+1:index+n,1) + nu;
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