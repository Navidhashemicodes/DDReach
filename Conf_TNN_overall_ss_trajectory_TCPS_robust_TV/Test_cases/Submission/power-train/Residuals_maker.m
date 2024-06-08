function residual =  Residuals_maker(Input_Data, Output_Data, net , delta, Num)


numdata = Num* floor((1+delta)/(1-delta));
thelen = size(Input_Data,2);
indexha = randperm(thelen, numdata);
Input_Data = Input_Data(:, indexha);
Output_Data = Output_Data(:, indexha);



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
residual = gather(abs( H - Output_Data));

end


