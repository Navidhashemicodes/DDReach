% clear all
% clc
% close all
% 
% D = mvnrnd(0,1,1000000000);
% histogram(D, 'NumBins' , 5000, 'DisplayStyle', 'stairs', 'EdgeColor',  'red', 'Normalization', 'probability', 'LineWidth', 1);
% 
% hold on
% for i = 1:1000
% 
%     C = sort(mvnrnd(0,1, 1000)');
%     plot(C(200), 0, 'x')
%     hold on
% 
% end

% clear all
% clc
% close all
% 
% alpha = 0.0001;
% m = 100000;
% ell = floor((1-alpha)*(m+1))+1;
% 
% hold on
% N = 100000;
% D= zeros(1, N );
% E = betarnd(ell, m+1-ell, [1, N]);
% parfor i = 1:N
%     DD = mvnrnd(0, 0.1, m)';
%     C = sort(DD);
%     D(i) = normcdf( C(ell ) , 0 , sqrt(0.1) );
% 
% end
% 
% figure
% 
% histogram(D, 'NumBins' , 500, 'DisplayStyle', 'stairs', 'EdgeColor',  'red', 'Normalization', 'probability', 'LineWidth', 1);
% 
% hold on
% 
% histogram(E, 'NumBins' , 500, 'DisplayStyle', 'stairs', 'EdgeColor',  'blue', 'Normalization', 'probability', 'LineWidth', 1);
% 
% cov(D)
% cov(E)
% mean(D)
% mean(E)


% clear all
% clc
% close all
% 
% rng(0)
% 
% N = 100000;
% D= zeros(1, N );
% 
% parfor i = 1:N
%     DD = mvnrnd(0, 10, 1);
%     D(i) = normcdf( DD , 0 , 10 );
% end
% 
% figure
% 
% histogram(D, 'NumBins' , 500, 'DisplayStyle', 'stairs', 'EdgeColor',  'red', 'Normalization', 'probability', 'LineWidth', 1);


% clear all
% clc
% close all
% 
% alpha = 0.5;
% m = 10000;
% ell = floor((1-alpha)*(m+1))+1;
% 
% hold on
% N = 100000;
% D= zeros(1, N );
% E = betarnd(ell, m+1-ell, [1, N]);
% parfor i = 1:N
%     DD = samplefrom(m);
%     C = sort(DD);
%     D(i) = cdfof( C(ell ) );
% end
% 
% figure
% 
% histogram(D, 'NumBins' , 500, 'DisplayStyle', 'stairs', 'EdgeColor',  'red', 'Normalization', 'probability', 'LineWidth', 1);
% 
% hold on
% 
% histogram(E, 'NumBins' , 500, 'DisplayStyle', 'stairs', 'EdgeColor',  'blue', 'Normalization', 'probability', 'LineWidth', 1);
% 
% cov(D)
% cov(E)
% mean(D)
% mean(E)



% clear all
% close all
% clc
% 
% 
% T = -10:0.01:10;
% Y = cdfof(T);
% plot( T(1:end-1) ,  Y(2:end)-Y(1:end-1))
% plot(T, Y );
% 
% [F1 , F2 ] = samplefrom(1);
% hold on
% 
% plot(F1, F2, 'x')


% 
% N = 100000000;
% [D,~] = samplefrom(N );
% histogram(D, 'NumBins' , 500, 'DisplayStyle', 'stairs', 'EdgeColor',  'blue', 'Normalization', 'probability', 'LineWidth', 1);








% function  [D, Yrand] = samplefrom(m)
% 
% 
% %%%%%% setting
% T1 = 0.001;
% T2 = 0.2;
% T3 = 0.3;
% T4 = 0.9;
% d = 5;
% 
% p1 = norminv(T1 , 0 , 1 );
% 
% a1 = 0.01; b1 = 0.1; c1 = T1 - a1*p1^2-b1*p1;
% p2 = ( -b1 + sqrt(b1^2 - 4*a1*(c1- T2)) ) / (2*a1) ;
% 
% a2 = 0.05;  b2 = T2 - a2*p2;
% p3 = ( T3 - b2) / a2;
% 
% p4 = p3+d;
% a3 = (T4-T3) / d;  b3 = T3 - a3*p3;
% 
% a4 = p4 - norminv(T4);
% %%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% Yrand = rand(1,m);
% 
% 
% for i=1:m
%     yrand = Yrand(i);
%     if yrand<= T1
%         D(i) = norminv(yrand , 0  ,1 );
%     elseif yrand>T1 && yrand<=T2
%         D(i) =  ( -b1 + sqrt(b1^2 - 4*a1*(c1- yrand)) ) / (2*a1) ;
%     elseif yrand>T2 && yrand<=T3
%         D(i) = (yrand-b2)/a2;
%     elseif yrand>T3 && yrand<=T4
%         D(i) = (yrand-b3)/a3;
%     elseif yrand>T4
%         D(i) = a4 + norminv(yrand);
%     end
% end
% 
% 
% 
% end
% 
% 
% 
% function  y = cdfof(the_rands)
% 
% %%%%%% setting
% 
% T1 = 0.001;
% T2 = 0.2;
% T3 = 0.3;
% T4 = 0.9;
% d = 5;
% 
% p1 = norminv(T1 , 0 , 1 );
% 
% a1 = 0.01; b1 = 0.1; c1 = T1 - a1*p1^2-b1*p1;
% p2 = ( -b1 + sqrt(b1^2 - 4*a1*(c1- T2)) ) / (2*a1) ;
% 
% a2 = 0.05;  b2 = T2 - a2*p2;
% p3 = ( T3 - b2) / a2;
% 
% p4 = p3+d;
% a3 = (T4-T3) / d;  b3 = T3 - a3*p3;
% 
% a4 = p4 - norminv(T4);
% 
% %%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% len  = length(the_rands);
% y = zeros(1,len);
% 
% for i=1:len
%     x = the_rands(i);
%     if x<=p1
%         y(i) = normcdf(x);
%     elseif x>p1 && x<=p2
%         y(i) = a1*x^2+b1*x+c1;
%     elseif x>p2 && x<=p3
%         y(i) = a2*x+b2;
%     elseif x>p3 && x<=p4
%         y(i) = a3*x + b3;
%     elseif x>p4
%         y(i) = normcdf(x-a4);
%     end
% end
% 
% 
% end


% clear all
% close all
% clc
% 
% T1 = 0.001;
% T2 = 0.2;
% T3 = 0.3;
% T4 = 0.9;
% d = 5;
% 
% p1 = norminv(T1 , 0 , 1 );
% 
% a1 = 0.01; b1 = +0.05; c1 = T1 - a1*p1^2-b1*p1;
% p2 = ( -b1 + sqrt(b1^2 - 4*a1*(c1- T2)) ) / (2*a1) ;
% 
% a2 = 0.05;  b2 = T2 - a2*p2;
% p3 = ( T3 - b2) / a2;
% 
% p4 = p3+d;
% a3 = (T4-T3) / d;  b3 = T3 - a3*p3;
% 
% a4 = p4 - norminv(T4);
% 
% 
% my_rand = linspace(p1 , p4, 10000);
% 
% y = zeros(1,10000);
% for i=1:10000
%     x = my_rand(i);
% 
%     if x>=p1 && x<p2
%         y(i) = a1*x^2+b1*x+c1;
%     elseif x>p2 && x<p3
%         y(i) = a2*x+b2;
%     elseif x>p3 && x<=p4
%         y(i) = a3*x + b3;
%     end
% end
% 
% 
% plot(my_rand , y)
% hold on 
% plot([p1, p2, p3, p4] , [0 0 0 0] , 'x')



% clear all
% clc
% close all
% load('s2s_Data_trajectory_train.mat')
% 
% for i = 1:1
% 
%     % F = reshape(Output_Data(:,floor(500*rand)+1), [5, 201]);
%     F = reshape(Output_Data(:,4600), [5, 201]);
%     for j=1:5
%         hold on
%         figure(j)
%         hold on
%         plot(F(j,:))
%         hold on
%     end
% end


clear

clc

addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))

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

load('s2s_Data_trajectory_train.mat')

for i = 1:1

    F = reshape(Output_Data(:,4400), [5, 201]);
    X = Output_Data(1:5, 1);
    Y = Output_Data(6:end, 1);
    Yh = [X ; NN(Net , X)];

    Fh = reshape(Yh, [5, 201]);

    for j=1:5
        hold on
        figure(j)
        hold on
        plot(F(j, :))
        hold on
        plot(Fh(j, :))
    end
end

% nn = 2; mm=2; pp=2; qq=2; rr=2;
% Number = nn*mm*pp*qq*rr;
% 
% for number=1:Number
% 
%     ii = floor((number-1)/(mm*pp*qq*rr))+1;
%     ji = number - (ii-1)*mm*pp*qq*rr      ;
%     jj = floor((ji    -1)/(pp*qq*rr)   )+1;
%     kj = ji     - (jj-1)*pp*qq*rr         ;
%     kk = floor((kj    -1)/(qq*rr)      )+1;
%     lk = kj     - (kk-1)*qq*rr            ;
%     ll = floor((lk    -1)/(rr)         )+1;
%     aa = lk     - (ll-1)*rr               ;
%     disp([ii, jj , kk , ll , aa]);
% end