clear all
clc
close all


pool = gcp('nocreate');

if ~isempty(pool)
    delete(pool);
end



lb = [-0.2 ; -0.2; -0.2; -0.2; -0.2; -0.2; 0; 0; 0; 0; 0; 0]; 
ub = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_Automatica_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));
timestep=0.1;
horizon = 50;


%%%%%%%%%%%%%%%%%


load('Main_network_Quad_stochasticsystem.mat')
Net.weights = net.weights;
Net.biases = net.biases;
dim = net.dims;
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'poslin'};
    Net.layers{i} = L ;
end


load('Rdist_0point11.mat')


cd Results
load('Quad_approx-star_084_rho_0point11_fdiv_abs_norobust_quantile_addLP.mat')
cd ..


Input_Data2 = gpuArray(Input_Data2); Output_Data2 = gpuArray(Output_Data2);
R_max_2 =  max_residual_newdist(Input_Data2, Output_Data2, Net , inv_Coefficients);

Output_Data2 = gather(Output_Data2);
Try_in1 = 10000;
decision = zeros(1, Try_in1);

experi_test = Output_Data2( :  , randperm(300000 , Try_in1) );

clear Output_Data2

parfor j=1: Try_in1

    decision(j) = check_contains(Star_sets , experi_test(:, j) );
    disp([ 'The ' num2str(j) '-th element of decision is : ' num2str(decision(j)) ])

end

s_cc = sum(decision);
conf_prob_cc = s_cc/Try_in1


L2 = length(R_max_2);
decision = zeros(1, L2);
parfor j=1: L2

    if R_max_2(1,j) <= maxim
        decision(j) = 1;
    end

end
s_cc2 = sum(decision);
conf_prob_cc2 = s_cc2/L2

experi_test = gpuArray(experi_test);
R_test_accordance = max_residual_newdist(experi_test(1:12,:), experi_test, s2s_model , inv_Coefficients);
decision = zeros(1,Try_in1);
parfor j=1: Try_in1

    if R_test_accordance(1,j) <= maxim
        decision(j) = 1;
    end

end
s_cc2_accordance= sum(decision);
conf_prob_cc2_accordance = s_cc2_accordance/Try_in1



function y = max_residual_newdist(Input_Data, Output_Data, net , inv_Coef)

n = size(Input_Data,1);
H = NN(net, Input_Data );
residual = abs( H - Output_Data(n+1:end,:));
residual_2 = residual./inv_Coef ;
y  = max(residual_2);

end

