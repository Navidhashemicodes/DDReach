function [inv_Coefficients,fval,exitflag] = Add_LP( Input_Data, Output_Data, net, inv_Coefficients, delta )

addpath(genpath('C:\gurobi1002'))

len = size(Input_Data,2);
n = size(Input_Data,1);
d = size(Output_Data);
H = NN(net, Input_Data );
residual = abs( H - Output_Data(n+1:end,:));

tau_init =  inv_Coefficients;
residual_2 = residual./tau_init ;
C_init  = max(residual_2);

loc = floor((len+1)*delta)+1;

if loc>len
    error('Not enough data for Conformal Inference')
end

RR = sort(C_init);

q_init = RR(1, loc);

residual = gather(residual);
C_init = gather(C_init);
q_init = gather(q_init);


A = zeros((d(1)-n)*d(2) , d(1)-n);
b = zeros((d(1)-n)*d(2) , 1   );
index = 0;
for j = 1:d(1)-n
    for i = 1:d(2)
        A(index+i , j) = -C_init(i);
        b(index+i , 1) = -residual(j, i);
    end
    index = index+d(2);
end
f = q_init*ones(d(1)-n,1);
[inv_Coefficients,fval,exitflag] = linprog(f,A,b);

end