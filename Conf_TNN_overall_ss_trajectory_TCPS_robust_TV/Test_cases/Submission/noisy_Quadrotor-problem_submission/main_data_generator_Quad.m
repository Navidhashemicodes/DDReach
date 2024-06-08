clear all
clc
close all


test = 1;
train= 1-test;
rng(1)

if test
    rng(0)
    lb = [-0.2 ; -0.2; -0.2; -0.2; -0.2; -0.2; 0; 0; 0; 0; 0; 0];
    ub = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];
    num_traj= 300000 ;
end

if train
    rng(1)
    lb = [-0.2 ; -0.2; -0.2; -0.2; -0.2; -0.2; 0; 0; 0; 0; 0; 0];
    ub = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];

    num_traj= 100000 ;
end

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))

load('Control.mat')
normalization=0;
timestep=0.1;
horizon = 50;

tic

% cov_m = 0.0001*eye(12);
cov_m = 0*eye(12);
avg_m = zeros(12,1);
cov_s = diag([0.05*ones(1,6), 0.01*ones(1,6)].^2);
avg_s = zeros(12,1);

[theInput, theOutput, maxmin] = Quad_12_nln_Datagenerator_ss(lb, ub, controller_nn, timestep, normalization, num_traj, horizon, avg_m, cov_m, avg_s, cov_s);

Data_run = toc;
Input_Data = theInput;
Output_Data = theOutput;

if test
    save('s2s_Data_trajectory_test_1.mat','Input_Data', 'Output_Data', 'maxmin', 'Data_run');
end

if train    
    save('s2s_Data_trajectory_train.mat','Input_Data', 'Output_Data', 'maxmin', 'Data_run');
end

clear all

