clear all
clc
close all

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));


load('Main_network_Quad_stochasticsystem.mat')
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


beta = 0.84;
dist_shift = 0.11;
dist_shift_char = '0point11';
f_type = 'abs';
analysis_type='approx-star';
Name = ['Quad_' analysis_type '_0', num2str(floor(beta*100)), '_rho_' dist_shift_char '_fdiv_' f_type '_robust_quantile_addLP'];


Center =  [ 0   ;  0  ;  0  ;  0  ;  0  ;  0  ; 0; 0; 0; 0; 0; 0]; 
epsilon = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];


load('s2s_Data_trajectory_train.mat')
In1 = Input_Data;
Ou1 = Output_Data;
clear Input_Data  Output_Data


load('Alpha_params_Quad_stochasticsystem.mat')
inv_Coefficients = (1./alpha_values(1:600))';
%%%%%%%%%%%%%%%%%


horizon = 50;

load('s2s_Data_trajectory_test.mat')
InputDatas_loc = gpuArray(Input_Data);
OutputDatas_loc = gpuArray(Output_Data);


L = size(OutputDatas_loc, 2);


%%%%%%%%%%%%%%%%


tic


delta= beta + dist_shift;


%%%%
numdata2 = 10000;
indexha2 = randperm(L, numdata2);
InputDatas_loc2 = In1(:, indexha2);
OutputDatas_loc2 = Ou1(:, indexha2);

%%%%

[inv_Coefficients,fval,exitflag] = Add_LP( InputDatas_loc2, OutputDatas_loc2, Net, inv_Coefficients, delta);

%%%%

Lp_time = toc;



tic

numdata = 10000;
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


nn = 2;
mm = 2;
pp = 2;
qq = 2;
rr = 2;
ss = 2;

Number = nn*mm*pp*qq*rr*ss;
lb = [-0.2 ; -0.2; -0.2; -0.2; -0.2; -0.2; 0; 0; 0; 0; 0; 0];
ub = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];
Grids1 = linspace(lb(1), ub(1), nn+1) ;
Grids2 = linspace(lb(2), ub(2), mm+1) ; 
Grids3 = linspace(lb(3), ub(3), pp+1) ;
Grids4 = linspace(lb(4), ub(4), qq+1) ;
Grids5 = linspace(lb(5), ub(5), rr+1) ; 
Grids6 = linspace(lb(6), ub(6), ss+1) ;
A = cell(1, Number)                   ;
Reachability_time = cell(1, Number)   ;
Star_sets_pre = cell(1,Number);
Star_sets     = cell(1,Number);

parfor number=1:Number
    tic;
    ii = floor((number-1)/(mm*pp*qq*rr*ss))+1;
    ji = number - (ii-1)*mm*pp*qq*rr*ss      ;
    jj = floor((ji    -1)/(pp*qq*rr*ss)   )+1;
    kj = ji     - (jj-1)*pp*qq*rr*ss         ;
    kk = floor((kj    -1)/(qq*rr*ss)      )+1;
    lk = kj     - (kk-1)*qq*rr*ss            ;
    ll = floor((lk    -1)/(rr*ss)         )+1;
    al = lk     - (ll-1)*rr*ss               ;
    aa = floor((al    -1)/(ss)            )+1;
    bb = al     - (aa-1)*ss                  ;
    Lb = [Grids1(ii)  ;Grids2(jj)  ;Grids3(kk)  ;Grids4(ll)  ;Grids5(aa)  ;Grids6(bb)  ; 0; 0; 0; 0; 0; 0];
    Ub = [Grids1(ii+1);Grids2(jj+1);Grids3(kk+1);Grids4(ll+1);Grids5(aa+1);Grids6(bb+1); 0; 0; 0; 0; 0; 0];
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