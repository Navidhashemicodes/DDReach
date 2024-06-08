function [conf_max , inv_Coef] = Conf_apply_nostack_max(Input_Data, Output_Data, net, delta)

len = size(Input_Data,2);
n = size(Input_Data,1);
d = size(Output_Data);
R = zeros(d);
H=gpuArray(R);
horizon = (d(1)/n)-1;
H(1:n, :) = Input_Data; 
for i=1:horizon
    index = i*n;
    H(index+1:index+n , :) = NN(net, H(index-n+1:index, : ) );
end

loc = floor((len+1)*delta)+1;

if loc>len
    error('Not enough data for Conformal Inference')
end

residual = abs( H(n+1:end,:) - Output_Data(n+1:end,:));
inv_Coef = max(residual')';


residual_2 = residual./inv_Coef ;
residual_max  = max(residual_2);
RR = sort(residual_max);

conf_max = RR(1,loc);


end