clear all
clc
close all


lb = [-0.5 ;  -0.5];
ub = [ 0.5 ;   0.5];


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));




cd Results
load('super_nonlinear_result_addLP.mat')
cd ..


nn = 20;
mm = 20;

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

Lb1 = Lb; Ub1 = Ub; 
clear Lbb Lb Ubb Ub


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd Results
load('super_nonlinear_result_MSE_addLP.mat')
cd ..


nn = 20;
mm = 20;

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

Lb2 = Lb; Ub2 = Ub; 
clear Lbb Lb Ubb Ub



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd Results
load('super_nonlinear_result_addLP_Linear_Lars.mat')
cd ..


nn = 20;
mm = 20;

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

Lb3 = Lb; Ub3 = Ub; 
clear Lbb Lb Ubb Ub


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load('s2s_Data_trajectory_test.mat')

D = Output_Data;


figure(2)
box on;

t = 0:50;


% Create a tiled layout
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot results over time
for i = 1:n
    nexttile;

    hold on;
    plot(t, Lb1(i, :), '-blue', 'LineWidth', 1);
    plot(t, Ub1(i, :), '-blue', 'LineWidth', 1);
    plot(t, Lb2(i, :), '-red', 'LineWidth', 1);
    plot(t, Ub2(i, :), '-red', 'LineWidth', 1);
    plot(t, Lb3(i, :), '-black', 'LineWidth', 1);
    plot(t, Ub3(i, :), '-black', 'LineWidth', 1);
    hold on;

    box on;
    set(gca, 'LineWidth', 3, 'FontSize', 12);  % Set axis box line width
    
    T = D(t*n+i, :);

    shaded_var_color(T, 'green')

    xticks([0,10,20,30,40,50]);  % Set desired tick positions
    xticklabels([0,10,20,30,40,50]);
    xlim([0,50])
    hold off;
end
