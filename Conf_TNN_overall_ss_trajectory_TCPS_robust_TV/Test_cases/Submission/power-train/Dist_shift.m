clear all
clc
close all


lb = [0.9*0.725 ; 0.9*14.7 ; 0.9*0.5455 ; 0 ; 1];
ub = [1.1*0.725 ; 1.1*14.7 ; 1.1*0.5455 ; 0 ; 1];
num_traj= 2000;

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust\src'))
normalization=0;
timestep=0.1;
simTime = 20;



load('s2s_Data_trajectory_test.mat')
Input_Data1 = Input_Data;
Output_Data1 = Output_Data;




%%%% Rdist0point5
%%% The noise is cov = 0.0105^2
load('Rdist_0point09.mat')

Input_Data2 = gpuArray(Input_Data2);
Output_Data2 = gpuArray(Output_Data2);



load('Main_network_PT.mat')
Net.weights = net.weights;
Net.biases = net.biases;
dim = net.dims;
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'poslin'};
    Net.layers{i} = L ;
end


cd Results
load('PT_exact-star_094_rho_0point5_fdiv_abs_robust_quantile_addLP.mat')
cd ..

Input_Data1 = gpuArray(Input_Data1); Output_Data1 = gpuArray(Output_Data1);
Input_Data2 = gpuArray(Input_Data2); Output_Data2 = gpuArray(Output_Data2);
R_max_1 =  max_residual_newdist(Input_Data1, Output_Data1, Net , inv_Coefficients);
R_max_2 =  max_residual_newdist(Input_Data2, Output_Data2, Net , inv_Coefficients);

clear Input_Data1 Input_Data2  Output_Data1

%%%%%  Dist_shift_estimator

UB = 1.1*max(R_max_2);
numbin = 50;
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


