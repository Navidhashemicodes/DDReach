function  Net  = Trapezius_maker_ss(model, T)

system_dim = size(model.biases{end},1);



ell_m = length(model.layers);
weights_m = model.weights;
biases_m  = model.biases;
layers_m  = model.layers;

for k = 1:T
    index = (k-1)*(ell_m);
    if k==1
        Net.weights{index+1} = sparse([eye(system_dim) ; weights_m{1}]);
        Net.biases{index+1} = sparse([zeros(system_dim,1) ; biases_m{1}]);
    else
        Net.weights{index+1} = sparse(blkdiag(eye((k-1)*system_dim) , [weights_m{end} ; weights_m{1}*weights_m{end}]));
        Net.biases{index+1} = sparse([zeros((k-1)*system_dim ,1)  ;  biases_m{end}   ;  weights_m{1}*biases_m{end}+biases_m{1} ]);
    end
    clear LL
    LL = cell(k*system_dim+size(biases_m{1},1),1);   %%% our structure to present a FFNN is different from the public structures, we present weights, biases and we also
    %%% represent the layers with a char array, that contains the name of activations, we further emply str2func() to make
    %%% the function from this characters. the parameter LL is introduced for this purpose
    LL(1:k*system_dim) = {'purelin'};
    LL(k*system_dim+1:end) = layers_m{1};
    Net.layers{index+1} = LL ;
    
    for j = 2:ell_m
        Net.weights{index+j} = sparse(blkdiag(eye(k*system_dim) , weights_m{j}));
        Net.biases{index+j} = sparse([zeros(k*system_dim,1) ; biases_m{j} ]) ;
        
        clear LL
        LL = cell(k*system_dim+size(biases_m{j},1),1);
        LL(1:k*system_dim) = {'purelin'};
        LL(k*system_dim+1:end) = layers_m{j};
        Net.layers{index+j} = LL ;
        
    end    
end
index = T*(ell_m);
Net.weights{index+1} = sparse(blkdiag(eye(T*system_dim) , weights_m{end}));
Net.biases{index+1} = sparse([zeros(k*system_dim,1) ; biases_m{end} ]) ;

end