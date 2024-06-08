function [q , tau] = Conf_apply_nostack_Bilinear_max(Input_Data, Output_Data, net, delta)

addpath(genpath('C:\gurobi1002'))

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
residual = abs( H(n+1:end,:) - Output_Data(n+1:end,:));


loc = floor((len+1)*delta)+1;

if loc>len
    error('Not enough data for Conformal Inference')
end

tau_init =  max(residual')';
residual_2 = residual./tau_init ;
C_init  = max(residual_2);
RR = sort(C_init);

q_init = RR(1,loc);
fval_init = q_init*ones(1,d(1)-n)*tau_init;

dd = true;
epsi =0.00001;
number = 0;

residual = gather(residual);
C_init = gather(C_init);
q_init = gather(q_init);
fval_init = gather(fval_init);

while dd

    number = number+1

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
    [tau_1,fval_1,exitflag] = linprog(f,A,b);

    if exitflag == 1
        residual_2 = residual./tau_1 ;
        C_1  = max(residual_2);
        RR = sort(C_1);
        q_1  = RR(1,loc);
        if abs( fval_init - fval_1 ) < epsi
            tau = tau_1;
            q  = q_1;
            dd = false;
        else
            q_init = q_1;
            C_init = C_1;
            fval_init = fval_1;
        end
    else
        disp(['exitflag is qual to: '  num2str(exitflag)])
        error('infeasible optimization')
    end



end


end