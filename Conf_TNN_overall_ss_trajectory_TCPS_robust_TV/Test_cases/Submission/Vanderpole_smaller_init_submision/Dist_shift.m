clear all
clc
close all


lb = [ -1.2  ; -1.2  ];
ub = [ -1.195 ; -1.195 ];

num_traj= 300000;

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
normalization=0;
timestep=0.02;
horizon = 50;


load('Rdist_1.mat')


%%%% Rdist0point085
tic

avg_m = zeros(2,1);
cov_m = 0*eye(2);
avg_s = zeros(2,1);
cov_s = 0.019*eye(2);

[theInput, theOutput, maxmin] = vdp_nln_Datagenerator_ss(lb, ub, timestep, normalization, num_traj , horizon, avg_m, cov_m, avg_s, cov_s) ;

Data_run = toc;


Input_Data2 = theInput;
Output_Data2 = theOutput;

Input_Data2 = gpuArray(Input_Data2);
Output_Data2 = gpuArray(Output_Data2);


load('Main_network_vdp_small_init_stochasticsystem.mat')
Net.weights = net.weights;
Net.biases = net.biases;
dim = net.dims;
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'poslin'};
    Net.layers{i} = L ;
end



cd Results
load( 'VDP_exact-star_077_rho_0point225_fdiv_abs_robust_quantile_addLP.mat' )
cd ..


Input_Data1 = gpuArray(Input_Data1); Output_Data1 = gpuArray(Output_Data1);
Input_Data2 = gpuArray(Input_Data2); Output_Data2 = gpuArray(Output_Data2);
R_max_1 =  max_residual_newdist(Input_Data1, Output_Data1, Net , inv_Coefficients);
R_max_2 =  max_residual_newdist(Input_Data2, Output_Data2, Net , inv_Coefficients);

clear Input_Data1 Input_Data2  Output_Data1

%%%%%  Dist_shift_estimator

UB = 1.1*max(R_max_2);
numbin = 1000;
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