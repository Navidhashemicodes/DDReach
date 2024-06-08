function inv_Coef= normalizer_generator(Input_Data, Output_Data, net)

n = size(Input_Data,1);
H = NN(net, Input_Data );


residual = abs( H - Output_Data(n+1:end,:));

inv_Coef = max(residual,[],2);


end