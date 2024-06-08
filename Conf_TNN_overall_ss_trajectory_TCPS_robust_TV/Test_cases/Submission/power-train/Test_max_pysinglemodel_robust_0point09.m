clear all
clc
close all

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));


load('Main_network_PT.mat')
Net.weights = net.weights;
Net.biases = net.biases;
dim = net.dims;
n = dim(1);
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'poslin'};
    Net.layers{i} = L ;
end


%%%%% be careful about the number of cells per parameter. It should be
%%%%% consistant with the STL specification you will introduce later,


beta = 0.90;
dist_shift = 0.09;
dist_shift_char = '0point09';
f_type = 'abs';
analysis_type='exact-star';
Name = ['PT_' analysis_type '_0', num2str(floor(beta*100)), '_rho_' dist_shift_char '_fdiv_' f_type '_robust_quantile_addLP'];

lb = [0.9*0.725 ; 0.9*14.7 ; 0.9*0.5455 ; 0 ; 1];
ub = [1.1*0.725 ; 1.1*14.7 ; 1.1*0.5455 ; 0 ; 1];
Center =  (lb+ub )/2; 
epsilon = (ub-lb )/2;


load('s2s_Data_trajectory_train.mat')
In1 = Input_Data;
Ou1 = Output_Data;
clear Input_Data  Output_Data


load('Alpha_params_PT.mat')
inv_Coefficients = (1./alpha_values(1:1000))';
%%%%%%%%%%%%%%%%%


horizon = 200;

load('s2s_Data_trajectory_test.mat')
InputDatas_loc = gpuArray(Input_Data);
OutputDatas_loc = gpuArray(Output_Data);


L = size(OutputDatas_loc, 2);


%%%%%%%%%%%%%%%%


tic


delta= beta + dist_shift;


%%%%
numdata2 = 1000;
indexha2 = randperm(L, numdata2);
InputDatas_loc2 = In1(:, indexha2);
OutputDatas_loc2 = Ou1(:, indexha2);

%%%%

[inv_Coefficients,fval,exitflag] = Add_LP( InputDatas_loc2, OutputDatas_loc2, Net, inv_Coefficients, delta);

%%%%

Lp_time = toc;



tic

numdata = 1000;
indexha = randperm(L, numdata);
InputDatas_loc1 = InputDatas_loc(:, indexha);
OutputDatas_loc1 = OutputDatas_loc(:, indexha);

%%%%


maxim = Conf_apply_nostack(InputDatas_loc1, OutputDatas_loc1, Net, delta, inv_Coefficients);


Conformal_time = toc;


Conf_overall = gather(maxim*inv_Coefficients);


disp('The inflation size:')
disp(sum(Conf_overall))


Leng = length(Conf_overall);


H = Star();
H.V = [zeros(Leng,1) eye(Leng)];
H.C = zeros(1,Leng);
H.d = 0;
H.predicate_lb = -Conf_overall;
H.predicate_ub =  Conf_overall;
H.dim = Leng;
H.nVar = Leng;

Conf_overalls = Conf_overall;


nn = 1;
mm = 1;
pp = 1;

Number = nn*mm*pp;

Grids1 = linspace(lb(1), ub(1), nn+1) ;
Grids2 = linspace(lb(2), ub(2), mm+1) ; 
Grids3 = linspace(lb(3), ub(3), pp+1) ;

A = cell(1, Number)                   ;
Reachability_time = cell(1, Number)   ;
Star_sets_pre = cell(1,Number);
Star_sets     = cell(1,Number);

parfor number=1:Number
    tic;
    ii = floor((number-1)/(mm*pp))+1;
    ji = number - (ii-1)*mm*pp      ;
    jj = floor((ji    -1)/(pp)   )+1;
    kk = ji     - (jj-1)*pp         ;
    
    
    Lb = [Grids1(ii)  ;Grids2(jj)  ;Grids3(kk)  ; 0 ; 1 ];
    Ub = [Grids1(ii+1);Grids2(jj+1);Grids3(kk+1); 0 ; 1 ];

    Star_sets_pre{number} = ReLUplex_Reachability_ss(   0.5*(Lb + Ub), 0.5*(-Lb + Ub), Net, analysis_type);


    lenS = length(Star_sets_pre{number});
    for cc=1:lenS
        Star_sets{number}(cc) = Sum( Star_sets_pre{number}(cc), H );
    end
    Reachability_time{number} = toc;

    A{number} =Star_sets{number};
end

cd Results
save( Name, 'Conf_overalls','Conformal_time', 'Star_sets', 'Reachability_time' , 'Lp_time', 'inv_Coefficients', 'maxim', 'A')
cd ..