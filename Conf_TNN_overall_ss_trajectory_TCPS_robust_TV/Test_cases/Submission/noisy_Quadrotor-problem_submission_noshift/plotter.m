% clear all
% clc
% close all




lb = [-0.2 ; -0.2; -0.2; -0.2; -0.2; -0.2; 0; 0; 0; 0; 0; 0]; 
ub = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));
timestep=0.1;
horizon = 50;


% load('certificate.mat')


nn = 2;
mm = 2;
pp = 2;
qq = 2;
rr = 2;
ss = 2;

Number = nn*mm*pp*qq*rr*ss;

n=12;

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

t = 0:50;


% Create a tiled layout
tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot results over time
for i = 1:n
    nexttile;

    hold on;
    plot(t, Lb(i, :), '-blue', 'LineWidth', 1);
    plot(t, Ub(i, :), '-blue', 'LineWidth', 1);
    hold on;

    box on;
    set(gca, 'LineWidth', 3, 'FontSize', 12);  % Set axis box line width


    xticks([0,10,20,30,40,50]);  % Set desired tick positions
    xticklabels([0,10,20,30,40,50]);
    xlim([0,50])
    hold off;
end
