clear all
clc
close all


lb = [-0.2 ; -0.2; -0.2; -0.2; -0.2; -0.2; 0; 0; 0; 0; 0; 0]; %#ok<UNRCH>
ub = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];
num_traj= 200000;

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust\src'))
load('Control.mat')
normalization=0;
timestep=0.1;
horizon = 50;



load('Rdist_1.mat')


%%%% Rdist0point24
% tic
% 
% cov_m = 0*eye(12);
% avg_m = zeros(12,1);
% cov_s = 1.35*diag([0.05*ones(1,6), 0.01*ones(1,6)].^2);
% avg_s = zeros(12,1);
% 
% [theInput, theOutput, maxmin] = Quad_12_nln_Datagenerator_ss(lb, ub, controller_nn, timestep, normalization, num_traj, horizon, avg_m, cov_m, avg_s, cov_s);


%%%% Rdist0point11
tic

cov_m = 0*eye(12);
avg_m = zeros(12,1);
cov_s = 1.15*diag([0.05*ones(1,6), 0.01*ones(1,6)].^2);
avg_s = zeros(12,1);

[theInput, theOutput, maxmin] = Quad_12_nln_Datagenerator_ss(lb, ub, controller_nn, timestep, normalization, num_traj, horizon, avg_m, cov_m, avg_s, cov_s);


Input_Data2 = theInput;
Output_Data2 = theOutput;

Input_Data2 = gpuArray(Input_Data2);
Output_Data2 = gpuArray(Output_Data2);



load('Main_network_Quad_stochasticsystem.mat')
Net.weights = net.weights;
Net.biases = net.biases;
dim = net.dims;
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'poslin'};
    Net.layers{i} = L ;
end


cd Results
load('Quad_approx-star_071_rho_0point24_fdiv_abs_robust_quantile_addLP.mat')
cd ..

Input_Data1 = gpuArray(Input_Data1); Output_Data1 = gpuArray(Output_Data1);
Input_Data2 = gpuArray(Input_Data2); Output_Data2 = gpuArray(Output_Data2);
R_max_1 =  max_residual_newdist(Input_Data1, Output_Data1, Net , inv_Coefficients);
R_max_2 =  max_residual_newdist(Input_Data2, Output_Data2, Net , inv_Coefficients);

clear Input_Data1 Input_Data2  Output_Data1

%%%%%  Dist_shift_estimator

UB = 1.1*max(R_max_2);
numbin = 500;
figure(1)
f1 = histogram(R_max_1, 'BinLimits' , [0,UB] , 'NumBins' , numbin, 'FaceAlpha', 0.3 , 'FaceColor' , 'red',  'EdgeAlpha', 0.3, 'EdgeColor',  'red', 'Normalization', 'probability');
hold on
f2 = histogram(R_max_2, 'BinLimits' , [0,UB] , 'NumBins' , numbin, 'FaceAlpha', 0.3 , 'FaceColor' , 'blue',  'EdgeAlpha', 0.3, 'EdgeColor',  'blue', 'Normalization', 'probability');

P = f1.Values;
Q = f2.Values;
dist = 0.5*sum( abs(P-Q) )





function y = max_residual_newdist(Input_Data, Output_Data, net , inv_Coef)

n = size(Input_Data,1);
H = NN(net, Input_Data );
residual = abs( H - Output_Data(n+1:end,:));
residual_2 = residual./inv_Coef ;
y  = max(residual_2);

end


