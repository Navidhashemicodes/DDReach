function y=NN(Net,x)
%%%% As you see in this toolbox we present our own data structure for NN. We return a NN as a tuple (weights, biases, layers).
%%%%  1-weights: is a cell of weight elements over the NN.
%%%%  2- bises : is a cell of bias vector elements in  NN.
%%%%  3- layers: is char array, which contains the name of activation functions. we can easily convert this char arrays to function using str2fuc().

out = x;
for i = 1:length(Net.layers)
    
    pre = Net.weights{i}*out+Net.biases{i};
    len = size(pre,1);
    out = zeros(len,size(pre,2));
    f=Net.layers{i};
    for ii=1:len
        p = pre(ii,:);
        if strcmp(f(ii), 'poslin')
            z = poslin(p);
        elseif strcmp(f(ii), 'sigmoid')
            z = sigmoid(p);
        elseif strcmp(f(ii), 'tanh')
            z = tanh(p);
        elseif strcmp(f(ii), 'purelin')
            z = p;
        elseif strcmp(f(ii), 'purelin_sat')
            z = p;
        else 
            error('Wrong activation function.')
        end
        out(ii,:)=z;
    end
    
end

y = Net.weights{end}*out+Net.biases{end};

end