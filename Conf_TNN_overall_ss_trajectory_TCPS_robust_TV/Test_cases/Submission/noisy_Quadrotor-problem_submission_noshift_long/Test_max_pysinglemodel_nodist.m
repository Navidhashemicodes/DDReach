clear all
clc
close all

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))
addpath(genpath('C:\gurobi1002\win64\matlab'));
addpath(genpath('C:\Users\navid\Documents\nnv-master'));
addpath(genpath('C:\Program Files\Mosek'));

cd Model_center\
load('Main_network_Quad_stochasticsystem_0_long.mat')
net0 = net;
load('Main_network_Quad_stochasticsystem_1_long.mat')
net1 = net;
load('Main_network_Quad_stochasticsystem_2_long.mat')
net2 = net;
load('Main_network_Quad_stochasticsystem_3_long.mat')
net3 = net;
load('Main_network_Quad_stochasticsystem_4_long.mat')
net4 = net;
cd ..

n = 12;

W = net0.weights;
B = net0.biases;
dim = net.dims;
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'poslin'};
    Ls{i} = L ;
end

W_add{1} = net1.weights;
B_add{1} = net1.biases;
dim_add{1} = net1.dims;
W_add{2} = net2.weights;
B_add{2} = net2.biases;
dim_add{2} = net2.dims;
W_add{3} = net3.weights;
B_add{3} = net3.biases;
dim_add{3} = net3.dims;
W_add{4} = net4.weights;
B_add{4} = net4.biases;
dim_add{4} = net4.dims;

for j =1:4
thelen = j*100;
wadd = sparse(W_add{j}{1});
badd = sparse(B_add{j}{1});
B{end} = [B{end} ; badd+wadd*B{end}(end-(n-1):end , :)];
W{end} = blkdiag(speye( thelen*n) ,  wadd  ) * [ W{end} ; W{end}(end-(n-1):end , :) ];
num = length(W_add{j});
for i=2:num

    L = cell( thelen*n + dim_add{j}(i),1 );
    L(  1:thelen*n  ) = {'purelin'};
    L(  thelen*n+1:end  ) = {'poslin'};
    Ls{end+1} =  L;
    
    wadd = sparse(W_add{j}{i});
    badd = sparse(B_add{j}{i});
    W{end+1} = blkdiag(  speye(thelen*n)  ,   wadd  );
    B{end+1} = [sparse(thelen*n,1) ; badd]  ;
end
end





Net.weights = W;
Net.biases = B;
Net.layers = Ls;

%%%% be careful about the number of cells per parameter. It should be
%%%% consistant with the STL specification you will introduce later,

analysis_type = 'approx-star';



Center =  [ 0   ;  0  ;  0  ;  0  ;  0  ;  0  ; 0; 0; 0; 0; 0; 0]; 
epsilon = [ 0.2 ;  0.2;  0.2;  0.2;  0.2;  0.2; 0; 0; 0; 0; 0; 0];


cd Data_center\
load('Stochastic_train_quad_dataset_long2.mat')
cd ..

In1 = X;
Ou1 = [X;Y];
clear X  Y

cd Model_center\
load('Alpha_params_Quad_stochasticsystem_0_long.mat')
inv_Coefficients = (1./alpha_values)';
load('Alpha_params_Quad_stochasticsystem_1_long.mat')
inv_Coefficients = [inv_Coefficients ; (1./alpha_values)'];
load('Alpha_params_Quad_stochasticsystem_2_long.mat')
inv_Coefficients = [inv_Coefficients ; (1./alpha_values)'];
load('Alpha_params_Quad_stochasticsystem_3_long.mat')
inv_Coefficients = [inv_Coefficients ; (1./alpha_values)'];
load('Alpha_params_Quad_stochasticsystem_4_long.mat')
inv_Coefficients = [inv_Coefficients ; (1./alpha_values)'];
cd ..

% inv_Coefficients = normalizer_generator(In1 , Ou1, Net);



%%%%%%%%%%%%%%%%%



horizon = 500;

cd Data_center\
load('Input2.mat')
load('Output2.mat')
InputDatas_loc = gpuArray(In2);
OutputDatas_loc = gpuArray(Out2);
clear In2 Out2

L = size(Ou1, 2);


% tic


delta= 0.995;


% %%%%
% numdata2 = 200;
% indexha2 = randperm(L, numdata2);
% InputDatas_loc2 = In1(:, indexha2);
% OutputDatas_loc2 = Ou1(:, indexha2);
% 
% %%%%
% 
% [inv_Coefficients,fval,exitflag] = Add_LP( InputDatas_loc2, OutputDatas_loc2, Net, inv_Coefficients, delta);
% 
% %%%%
% 
% Lp_time = toc;

tic

L = size(OutputDatas_loc, 2);
numdata = 8000;
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
Star_setss     = cell(1,Number);
% poolobj = gcp('nocreate');
% if isempty(poolobj)
%     parpool('local', 10);
% else
%     poolobj.NumWorkers = 10;
% end
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
        Star_setss{number}(cc) = Sum( Star_sets_pre{number}(cc), H );
    end
    Reachability_time{number} = toc;

    A{number} =Star_setss{number};
end

save('GGG.mat', 'Star_setss')


for i=1:Number
    Star_sets{1}(i) = Star_setss{i};
end

Reachability_time = sum([Reachability_time{:}]);


Hor = size(  Star_sets{1}(1).V  ,  1  );
Horiz = (Hor/n)-1;


parfor number=1:Number
    Box = Overall_Box( Star_sets{1}(number), eye(Hor), zeros(Hor,1));
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

D = Ou1;

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
    
    T = D(t*n+i, :);

    shaded_var_color(T, 'green')

    xticks([0,10,20,30,40,50]);  % Set desired tick positions
    xticklabels([0,10,20,30,40,50]);
    xlim([0,50])
    hold off;
end




% Output_Data1 = gather(Output_Data1);
% Try_in = 30000;
% decision = zeros(1, Try_in);
% experi_test = Output_Data1( :  , randperm(300000 , Try_in) );
% 
% 
% parfor j=1: Try_in
% 
%     Decision = check_contains_ayahast(Star_sets , experi_test(13:end, j));
% 
%     if Decision == 1
%         decision(j) = 1;
%     else
%         disp('ayahast mentioned No')
%         decision(j) = check_contains_agenist(Star_sets , experi_test(13:end, j) , n);
%         if decision(j)==1
%             disp('agenist is in disagreement and found it')
%         else
%             disp('agenist confirmed it')
%         end
%     end
%     disp([ 'The ' num2str(j) '-th element of decision is : ' num2str(decision(j)) ])
% 
% end
% clear Output_Data1
% s_cc = sum(decision);
% conf_prob_cc = s_cc/Try_in
% 
% 
% 
% experi_test = gpuArray(experi_test);
% R_test_accordance = max_residual_dist(experi_test(1:12,:), experi_test, Net , inv_Coefficients);
% s_cc2_accordance= sum(R_test_accordance <= maxim);
% conf_prob_cc2_accordance = s_cc2_accordance/Try_in
% 
% clear  In1  InputDatas_loc  InputDatas_loc1 Ou1  OutputDatas_loc  OutputDatas_loc1 
% 
% 
% R_test = gather(R_test);
% R_test_accordance = gather(R_test_accordance);
% s_cc2 = gather(s_cc2);
% s_cc2_accordance = gather(s_cc2_accordance);
% maxim = gather(maxim);
% H= gather(H);
% experi_test = gather(experi_test);
% conf_prob_cc2 = gather(conf_prob_cc2);
% conf_prob_cc2_accordance = gather(conf_prob_cc2_accordance);