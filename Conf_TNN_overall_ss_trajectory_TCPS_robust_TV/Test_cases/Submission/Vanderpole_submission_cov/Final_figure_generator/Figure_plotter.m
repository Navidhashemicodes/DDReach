clear all
close all
clc


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\Test_cases\Submission\Vanderpole_submission_cov'))
addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));

load('Rdist_1.mat')
Out1 = Output_Data1;
clear Input_Data1 Output_Data1

load('Rdist_0point085.mat')
Out2 = Output_Data2;
clear Input_Data2 Output_Data2


%%%%%%%%%%%%%


lb = [-1.2 ; -1.2];
ub = [-1   ; -1  ];

load('VDP_stars.mat')


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
Lb_bound = Lb; Ub_bound = Ub;
clear Lb Ub





figure(2)
box on;

t = 0:50;


% Create a tiled layout
ttt = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:n
    nexttile;

    hold on;

    T2 = Out2(t*n+i, :);
    shaded_var_color(T2, 'yellow')
    T1 = Out1(t*n+i, :);
    shaded_var_color(T1, 'green') 

    hold on;
    plot(t, Lb_bound(i, :)  , '-black'  , 'LineWidth', 2);
    plot(t, Ub_bound(i, :)  , '-black'  , 'LineWidth', 2);

    box on;
    set(gca, 'LineWidth', 3, 'FontSize', 12);  % Set axis box line width


    xticks([0,10,20,30,40,50]);  % Set desired tick positions
    xticklabels([0,10,20,30,40,50]);
    xlim([0,50])
    hold off;
end
