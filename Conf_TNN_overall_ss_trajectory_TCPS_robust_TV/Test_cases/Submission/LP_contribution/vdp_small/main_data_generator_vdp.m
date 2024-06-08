clear all
clc
close all

test = 1;
train= 1-test;
rng(1)

if test
    rng(0)
    lb = [ -1.2  ; -1.2  ];
    ub = [ -1.195 ; -1.195 ];
    num_traj= 40000 ;
end

if train
    rng(1)
    lb = [ -1.2  ; -1.2  ];
    ub = [ -1.195 ; -1.195 ];

    num_traj= 100000 ;
end

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
normalization=0;
timestep=0.02;
horizon = 50;
tic

avg_m = zeros(2,1);
% cov_m = 0.0001*eye(2);
cov_m = 0*eye(2);
avg_s = zeros(2,1);
cov_s = 0.01*eye(2);

[theInput, theOutput, maxmin] = vdp_nln_Datagenerator_ss(lb, ub, timestep, normalization, num_traj, horizon, avg_m, cov_m, avg_s, cov_s);
Data_run = toc;
Input_Data = theInput;
Output_Data = theOutput;
if test
    save('s2s_Data_trajectory_test.mat','Input_Data', 'Output_Data', 'maxmin', 'Data_run');
end

if train
    
    save('s2s_Data_trajectory_train.mat','Input_Data', 'Output_Data', 'maxmin', 'Data_run');
    
end


clear all
