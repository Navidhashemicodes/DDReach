function y = max_residual_dist(Input_Data, Output_Data, net , inv_Coef)

n = size(Input_Data,1);
H = NN(net, Input_Data );
residual = abs( H - Output_Data(n+1:end,:));
residual_2 = residual./inv_Coef ;
y  = max(residual_2);

end