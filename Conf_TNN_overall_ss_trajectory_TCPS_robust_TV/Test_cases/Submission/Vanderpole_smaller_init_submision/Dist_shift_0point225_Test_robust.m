clear all
clc
close all

lb = [ -1.2  ; -1.2  ];
ub = [ -1.195 ; -1.195 ];

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));
timestep = 0.02;
horizon = 50;


%%%%%%%%%%%%%%%%%


load('Rdist_1.mat')
num_traj= size(Input_Data1 , 2);

load('Rdist_0point225.mat')


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
load('VDP_exact-star_077_rho_0point225_fdiv_abs_robust_quantile_addLP.mat')
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

%%%%%%%%%%%%%%%%%%%%%%%%%

clear f1 f2


nn = 1;
mm = 1;

Number = nn*mm;

n=2;

Hor = size(  Star_sets{1}(1).V  ,  1  );
Horiz = (Hor/n)-1;

parfor number=1:Number
    Box = Overall_Box( Star_sets{number}, eye(Hor), zeros(Hor,1));
    Lbb{number} = reshape( Box(:,1) ,  [ n , Horiz+1 ] );
    Ubb{number} = reshape( Box(:,2) ,  [ n , Horiz+1 ] );
end
Lb = zeros(n, Horiz+1);
Ub = zeros(n, Horiz+1);
Lb(:,1) = lb;
Ub(:,1) = ub;


for i=1:Horiz+1

    min1 = Lbb{1}(1,i);
    min2 = Lbb{1}(2,i);


    max1 = Ubb{1}(1,i);
    max2 = Ubb{1}(2,i);

    for j=2:Number
        min1 = min( min1 , Lbb{j}(1,i) );
        min2 = min( min2 , Lbb{j}(2,i) );

        max1 = max( max1 , Ubb{j}(1,i) );
        max2 = max( max2 , Ubb{j}(2,i) );
    end
    Lb(:,i+1) = [min1 ; min2];
    Ub(:,i+1) = [max1 ; max2];
end





figure(2)
box on;
D= Output_Data1;



t = 0:50;


% Create a tiled layout
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot results over time
for i = 1:n
    nexttile;


    T = D(t*n+i, :);
    shaded_var(T)
    
    hold on
    for ii=1:1000
        plot( t , T(  : , floor(300000*rand)+1 )'  )
        hold on
    end


    hold on;
    plot(t, Lb(i, :), '-black', 'LineWidth', 2);
    plot(t, Ub(i, :), '-black', 'LineWidth', 2);
    hold on;

    

    box on;
    set(gca, 'LineWidth', 3, 'FontSize', 12);  % Set axis box line width


    xticks([0,10,20,30,40,50]);  % Set desired tick positions
    xticklabels([0,10,20,30,40,50]);
    xlim([0,50])
    hold off;
end

clear D T







% L2 = length(R_max_2);
% decision = zeros(1, L2);
% parfor j=1: L2
% 
%     if R_max_2(1,j) <= maxim
%         decision(j) = 1;
%     end
% 
% end
% s_cc2 = sum(decision);
% conf_prob_cc2 = s_cc2/L2
% 
% 
% 
% 
% n=2;
% Output_Data2 = gather(Output_Data2);
% Try_in = 10000;
% decision = zeros(1, Try_in);
% experi_test = Output_Data2( :  , randperm(300000 , Try_in) );
% parfor j=1: Try_in
% 
%     Decision = check_contains_ayahast(Star_sets , experi_test(:, j));
% 
%     if Decision == 1
%         decision(j) = 1;
%     else
%         disp('ayahast mentioned No')
%         decision(j) = check_contains_agenist(Star_sets , experi_test(:, j) , n);
%         if decision(j)==1
%             disp('agenist is in disagreement and found it')
%         else
%             disp('agenist confirmed it')
%         end
%     end
%     disp([ 'The ' num2str(j) '-th element of decision is : ' num2str(decision(j)) ])
% 
% end
% 
% clear Output_Data2
% s_cc = sum(decision);
% conf_prob_cc = s_cc/Try_in
% 
% 
% experi_test = gpuArray(experi_test);
% R_test_accordance = max_residual_newdist(experi_test(1:2,:), experi_test, s2s_model , inv_Coefficients);
% decision = zeros(1,Try_in);
% parfor j=1: Try_in
% 
%     if R_test_accordance(1,j) <= maxim
%         decision(j) = 1;
%     end
% 
% end
% s_cc2_accordance= sum(decision);
% conf_prob_cc2_accordance = s_cc2_accordance/Try_in



function y = max_residual_newdist(Input_Data, Output_Data, net , inv_Coef)

n = size(Input_Data,1);
H = NN(net, Input_Data );
residual = abs( H - Output_Data(n+1:end,:));
residual_2 = residual./inv_Coef ;
y  = max(residual_2);

end


