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


load('Rdist_1.mat')
num_traj= size(Input_Data1 , 2);

load('Rdist_0point24.mat')

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



n=12;

Hor = size(  Star_sets{1}.V  ,  1  );
Horiz = (Hor/n)-1;

nn = 2;
mm = 2;
pp = 2;
qq = 2;
rr = 2;
ss = 2;
Number = nn*mm*pp*qq*rr*ss;

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
    min3 = Lbb{1}(3,i);
    min4 = Lbb{1}(4,i);
    min5 = Lbb{1}(5,i);
    min6 = Lbb{1}(6,i);
    min7 = Lbb{1}(7,i);
    min8 = Lbb{1}(8,i);
    min9 = Lbb{1}(9,i);
    min10 = Lbb{1}(10,i);
    min11 = Lbb{1}(11,i);
    min12 = Lbb{1}(12,i);


    max1 = Ubb{1}(1,i);
    max2 = Ubb{1}(2,i);
    max3 = Ubb{1}(3,i);
    max4 = Ubb{1}(4,i);
    max5 = Ubb{1}(5,i);
    max6 = Ubb{1}(6,i);
    max7 = Ubb{1}(7,i);
    max8 = Ubb{1}(8,i);
    max9 = Ubb{1}(9,i);
    max10 = Ubb{1}(10,i);
    max11 = Ubb{1}(11,i);
    max12 = Ubb{1}(12,i);

    for j=2:Number
        min1 = min( min1 , Lbb{j}(1,i) );
        min2 = min( min2 , Lbb{j}(2,i) );
        min3 = min( min3 , Lbb{j}(3,i) );
        min4 = min( min4 , Lbb{j}(4,i) );
        min5 = min( min5 , Lbb{j}(5,i) );
        min6 = min( min6 , Lbb{j}(6,i) );
        min7 = min( min7 , Lbb{j}(7,i) );
        min8 = min( min8 , Lbb{j}(8,i) );
        min9 = min( min9 , Lbb{j}(9,i) );
        min10 = min( min10 , Lbb{j}(10,i) );
        min11 = min( min11 , Lbb{j}(11,i) );
        min12 = min( min12 , Lbb{j}(12,i) );

        max1 = max( max1 , Ubb{j}(1,i) );
        max2 = max( max2 , Ubb{j}(2,i) );
        max3 = max( max3 , Ubb{j}(3,i) );
        max4 = max( max4 , Ubb{j}(4,i) );
        max5 = max( max5 , Ubb{j}(5,i) );
        max6 = max( max6 , Ubb{j}(6,i) );
        max7 = max( max7 , Ubb{j}(7,i) );
        max8 = max( max8 , Ubb{j}(8,i) );
        max9 = max( max9 , Ubb{j}(9,i) );
        max10 = max( max10 , Ubb{j}(10,i) );
        max11 = max( max11 , Ubb{j}(11,i) );
        max12 = max( max12 , Ubb{j}(12,i) );
    end
    Lb(:,i+1) = [min1 ; min2; min3 ; min4; min5 ; min6; min7 ; min8; min9 ; min10; min11 ; min12];
    Ub(:,i+1) = [max1 ; max2; max3 ; max4; max5 ; max6; max7 ; max8; max9 ; max10; max11 ; max12];
end



figure(2)
box on;
D= Output_Data2;



t = 0:50;


% Create a tiled layout
tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot results over time
for i = 1:n
    nexttile;


    T = D(t*n+i, :);
    shaded_var(T)
    
    hold on
    for ii=1:100
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


% Output_Data2 = gather(Output_Data2);
% Try_in1 = 10000;
% decision = zeros(1, Try_in1);
% 
% experi_test = Output_Data2( :  , randperm(300000 , Try_in1) );
% 
% 
% 
% parfor j=1: Try_in1
% 
%     decision(j) = check_contains(Star_sets , experi_test(:, j) );
%     disp([ 'The ' num2str(j) '-th element of decision is : ' num2str(decision(j)) ])
% 
% end
% clear Output_Data2
% s_cc = sum(decision);
% conf_prob_cc = s_cc/Try_in1
% 
% 
% 
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
% experi_test = gpuArray(experi_test);
% R_test_accordance = max_residual_newdist(experi_test(1:12,:), experi_test, s2s_model , inv_Coefficients);
% decision = zeros(1,Try_in1);
% parfor j=1: Try_in1
% 
%     if R_test_accordance(1,j) <= maxim
%         decision(j) = 1;
%     end
% 
% end
% s_cc2_accordance= sum(decision);
% conf_prob_cc2_accordance = s_cc2_accordance/Try_in1


function y = max_residual_newdist(Input_Data, Output_Data, net , inv_Coef)

n = size(Input_Data,1);
H = NN(net, Input_Data );
residual = abs( H - Output_Data(n+1:end,:));
residual_2 = residual./inv_Coef ;
y  = max(residual_2);

end


