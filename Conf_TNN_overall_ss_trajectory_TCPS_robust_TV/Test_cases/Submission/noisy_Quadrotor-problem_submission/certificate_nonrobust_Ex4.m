clear all
clc
close all


lb = [-0.2 ; -0.2; -0.2; -0.2; -0.2; -0.2; 0; 0; 0; 0; 0; 0]; 
ub = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));
timestep=0.1;
horizon = 50;


%%%%%%%%%%%%%%%%%



load('Rdist_0point11.mat')

clear Input_Data2

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
load('Quad_approx-star_071_rho_0point24_fdiv_abs_norobust_quantile_addLP.mat')
cd ..

Star_setss{1} = Star();
for i = 1:length(Star_sets )
    Star_setss{1}(i) = Star_sets{i};
end



Output_Data2 = gather(Output_Data2);
Try_in = 2000;
decision = zeros(1, Try_in);
experi_test = Output_Data2( :  , randperm(300000 , Try_in) );

n=12;

parfor j=1: Try_in

    Decision = check_contains_ayahast(Star_setss , experi_test(13:end, j));

    if Decision == 1
        decision(j) = 1;
    else
        disp('ayahast mentioned No')
        decision(j) = check_contains_agenist(Star_setss , experi_test(13:end, j) , n);
        if decision(j)==1
            disp('agenist is in disagreement and found it')
        else
            disp('agenist confirmed it')
        end
    end
    disp([ 'The ' num2str(j) '-th element of decision is : ' num2str(decision(j)) ])

end
clear Input_Data2 Output_Data2
s_cc = sum(decision);
conf_prob_cc = s_cc/Try_in



experi_test = gpuArray(experi_test);
R_test_accordance = max_residual_dist(experi_test(1:12,:), experi_test, Net , inv_Coefficients);
s_cc2_accordance= sum(R_test_accordance <= maxim);
conf_prob_cc2_accordance = s_cc2_accordance/Try_in



R_test_accordance = gather(R_test_accordance);
s_cc2_accordance = gather(s_cc2_accordance);
maxim = gather(maxim);
experi_test = gather(experi_test);
conf_prob_cc2_accordance = gather(conf_prob_cc2_accordance);