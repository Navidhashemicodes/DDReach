clear all
clc
close all


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_Automatica_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));
timestep = 0.02;
horizon = 50;

%%%%%%%%%%%%%%%%%

spmd
  gpuDevice( 1 + mod( labindex - 1, gpuDeviceCount ) )
end

%%%%%%%%%%%%%%%%%




load('Rdist_0point225.mat')
Input_Data2 = gpuArray(Input_Data) ; Output_Data2 = gpuArray( Output_Data);

clear Output_Data Input_Data

load('Main_network.mat')
Net.weights = net.weights;
Net.biases = net.biases;
dim = net.dims;
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'poslin'};
    Net.layers{i} = L ;
end
s2s_model = Net;


cd Results
load('VDP_exact-star_077_rho_0point225_fdiv_abs_trajectory_exact_max_pysinglemodel_norobust.mat')
cd ..

R_max_2 =  max_residual_newdist(Input_Data2, Output_Data2, s2s_model , inv_Coefficients);

clear Input_Data2 



L2 = length(R_max_2);
decision = zeros(1, L2);
parfor j=1: L2

    if R_max_2(1,j) <= maxim
        decision(j) = 1;
    end

end
s_cc2 = sum(decision);
conf_prob_cc2 = s_cc2/L2


n=2;
Output_Data2 = gather(Output_Data2);
Try_in = 10000;
decision = zeros(1, Try_in);
experi_test = Output_Data2( :  , randperm(300000 , Try_in) );


parfor j=1: Try_in

    Decision = check_contains_ayahast(Star_sets , experi_test(:, j));

    if Decision == 1
        decision(j) = 1;
    else
        disp('ayahast mentioned No')
        decision(j) = check_contains_agenist(Star_sets , experi_test(:, j) , n);
        if decision(j)==1
            disp('agenist is in disagreement and found it')
        else
            disp('agenist confirmed it')
        end
    end
    disp([ 'The ' num2str(j) '-th element of decision is : ' num2str(decision(j)) ])

end
clear Output_Data2
s_cc = sum(decision);
conf_prob_cc = s_cc/Try_in



experi_test = gpuArray(experi_test);
R_test_accordance = max_residual_newdist(experi_test(1:2,:), experi_test, s2s_model , inv_Coefficients);
decision = zeros(1,Try_in);
parfor j=1: Try_in

    if R_test_accordance(1,j) <= maxim
        decision(j) = 1;
    end

end
s_cc2_accordance= sum(decision);
conf_prob_cc2_accordance = s_cc2_accordance/Try_in





function y = max_residual_newdist(Input_Data, Output_Data, net , inv_Coef)

n = size(Input_Data,1);
d = size(Output_Data);
R = zeros(d);
H=gpuArray(R);
horizon = (d(1)/n)-1;
H(1:n, :) = Input_Data; 
for i=1:horizon
    index = i*n;
    H(index+1:index+n , :) = NN(net, H(index-n+1:index, : ) );
end

residual = abs( H(n+1:end,:) - Output_Data(n+1:end,:));

residual_2 = residual./inv_Coef ;
y  = max(residual_2);

end


