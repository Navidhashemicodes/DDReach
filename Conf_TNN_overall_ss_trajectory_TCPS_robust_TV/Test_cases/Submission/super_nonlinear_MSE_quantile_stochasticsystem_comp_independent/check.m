clear all

clc

close all


addpath(genpath('C:\Users\navid\Documents\MATLAB\MATLAB_prev\others\Files\CDC2023\Conf_TNN_overall_ss_trajectory_TCPS_robust_TV\src'))



load('Main_network.mat')
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
In1 = Input_Data;
Ou1 = Output_Data;
clear Input_Data  Output_Data

ind = floor(rand*100000)+1;
pred = NN(Net, In1(:,ind));

plot(Ou1(3:end,ind), 'blue')
hold on
plot(pred, 'red')