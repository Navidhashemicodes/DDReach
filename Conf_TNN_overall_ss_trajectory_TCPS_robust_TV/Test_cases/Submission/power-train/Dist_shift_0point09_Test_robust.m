clear all
clc
close all




lb = [0.9*0.725 ; 0.9*14.7 ; 0.9*0.5455 ; 0 ; 1];
ub = [1.1*0.725 ; 1.1*14.7 ; 1.1*0.5455 ; 0 ; 1];


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));
timestep=0.1;
horizon = 200;


%%%%%%%%%%%%%%%%%


load('s2s_Data_trajectory_test.mat')
Input_Data1 = Input_Data;
Output_Data1 = Output_Data;
num_traj= size(Input_Data1 , 2);

load('Rdist_0point09.mat')

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
load('PT_exact-star_090_rho_0point09_fdiv_abs_robust_quantile_addLP.mat')
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

%%%%%%%%%%%%%%%%%%%%%%%%%

clear f1 f2


n=5;

Hor = size(  Star_sets{1}(1).V  ,  1  );
Horiz = (Hor/n)-1;

nn = 1;
mm = 1;
pp = 1;

Number = nn*mm*pp;

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


    max1 = Ubb{1}(1,i);
    max2 = Ubb{1}(2,i);
    max3 = Ubb{1}(3,i);
    max4 = Ubb{1}(4,i);
    max5 = Ubb{1}(5,i);

    for j=2:Number
        min1 = min( min1 , Lbb{j}(1,i) );
        min2 = min( min2 , Lbb{j}(2,i) );
        min3 = min( min3 , Lbb{j}(3,i) );
        min4 = min( min4 , Lbb{j}(4,i) );
        min5 = min( min5 , Lbb{j}(5,i) );

        max1 = max( max1 , Ubb{j}(1,i) );
        max2 = max( max2 , Ubb{j}(2,i) );
        max3 = max( max3 , Ubb{j}(3,i) );
        max4 = max( max4 , Ubb{j}(4,i) );
        max5 = max( max5 , Ubb{j}(5,i) );
    end
    Lb(:,i+1) = [min1 ; min2; min3 ; min4; min5];
    Ub(:,i+1) = [max1 ; max2; max3 ; max4; max5];
end



figure(2)
box on;
D= Output_Data2;



t = 0:200;


% Create a tiled layout
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot results over time
for i = 1:n
    nexttile;


    T = D(t*n+i, :);
    shaded_var(T)
    
    hold on
    for ii=1:100
        plot( t , T(  : , floor(2000*rand)+1 )'  )
        hold on
    end


    hold on;
    plot(t, Lb(i, :), '-black', 'LineWidth', 2);
    plot(t, Ub(i, :), '-black', 'LineWidth', 2);
    hold on;

    

    box on;
    set(gca, 'LineWidth', 3, 'FontSize', 12);  % Set axis box line width


    xticks([0,10,20,30,40,50, 60,70,80,90,100,110, 120,130,140,150, 160, 170,180,190,200]);  % Set desired tick positions
    xticklabels([0,10,20,30,40,50, 60,70,80,90,100,110, 120,130,140,150, 160, 170,180,190,200]);
    xlim([0,200])
    hold off;
end

clear D T


function y = max_residual_newdist(Input_Data, Output_Data, net , inv_Coef)

n = size(Input_Data,1);
H = NN(net, Input_Data );
residual = abs( H - Output_Data(n+1:end,:));
residual_2 = residual./inv_Coef ;
y  = max(residual_2);

end
