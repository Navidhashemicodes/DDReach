function [conf_max , inv_Coef] = Conf_apply_nostack_CP(Input_Data, Output_Data, net, delta, delta_alpha)

len = size(Input_Data,2);
n = size(Input_Data,1);
d = size(Output_Data);
R = zeros(d(1)-n,d(2));
H=gpuArray(zeros(d));
horizon = (d(1)/n)-1;
H(1:n, :) = Input_Data; 
for i=1:horizon
    index = i*n;
    H(index+1:index+n , :) = NN(net, H(index-n+1:index, : ) );
end
residual = abs( H(n+1:end,:) - Output_Data(n+1:end,:));


for j=1:d(1)-n
    R(j,:) = sort(residual(j,:));
end

loc1 = floor((len+1)*delta_alpha)+1;
loc2 = floor((len+1)*delta)+1;

if max(loc1, loc2) > len
    error('Not enough data for Conformal Inference')
end
inv_Coef = R(:,loc1);


residual_2 = residual./inv_Coef ;
residual_max  = max(residual_2);
RR = sort(residual_max);

conf_max = RR(1,loc2);


end