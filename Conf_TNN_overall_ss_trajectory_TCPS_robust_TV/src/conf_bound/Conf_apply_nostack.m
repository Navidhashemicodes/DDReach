function conf_max = Conf_apply_nostack(Input_Data, Output_Data, net, delta, inv_Coef)

len = size(Input_Data,2);
n = size(Input_Data,1);
H = NN(net, Input_Data );

loc = floor((len+1)*delta)+1;

if loc>len
    error('Not enough data for Conformal Inference')
end

residual = abs( H - Output_Data(n+1:end,:));

residual_2 = residual./inv_Coef ;
residual_max  = max(residual_2);
RR = sort(residual_max);

conf_max = RR(1,loc);


end