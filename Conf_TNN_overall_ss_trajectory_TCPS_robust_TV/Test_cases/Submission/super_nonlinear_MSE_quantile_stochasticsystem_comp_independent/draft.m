% clear
% clc
% close all
% 
% N = 1;
% x = cell(1,N);
% y = cell(1,N);
% for j=1:N
% 
% 
% % x{j}(:,1)= -0.5+1*rand(2,1);
% x{j}(:,1)= zeros(2,1);
% avg = zeros(2,1);
% cov = 0.005*eye(2);
% nu = mvnrnd(avg , cov , 1 )';
% y{j}(:,1) = x{j}(:,1)+nu;
% for i=1:50
%     x{j}(:,i+1) = myf(x{j}(:,i));
%     avg = zeros(2,1);
%     cov = 0.001*eye(2);
%     nu = mvnrnd(avg , cov , 1 )';
%     y{j}(:,i+1) = x{j}(:,i+1)+nu;
% end
% 
% 
% hold on
% plot(reshape(y{j},[1,102]));
% 
% end
% 
% 
% 
% 
% function s = myf(s)
% x = s(1);
% y = s(2);
% x2 = 0.985*y+sin(0.5*x)-0.6*sin(x+y)-0.07;
% y2 = 0.985*x+cos(0.5*y)-0.6*cos(x+y)-0.07;
% s = [x2;y2];
% 
% end


load('s2s_Data_trajectory_test2.mat')

D = Output_Data;

t = 0:50;

n = 2;
i = 1;

T = D(t*n+i, :);

starting = [1,1,0.8];  %%% Light yellow
ending = [0,0.5,0];  %%% green
numbins = 100;
shaded_var_color_density(T, starting, ending, numbins)



