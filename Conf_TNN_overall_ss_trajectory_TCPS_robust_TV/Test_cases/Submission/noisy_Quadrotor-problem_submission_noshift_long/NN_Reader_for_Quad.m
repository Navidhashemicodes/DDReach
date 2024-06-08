function nn_controller =  NN_Reader_for_Quad(name)


file = fopen(name,'r');
file_data = fscanf(file,'%f');
no_of_inputs = file_data(1);
no_of_outputs = file_data(2);
no_of_hidden_layers = file_data(3);
network_structure = zeros(no_of_hidden_layers+1,1);
pointer = 4;
for i = 1:no_of_hidden_layers
    network_structure(i) = file_data(pointer);
    pointer = pointer + 1;
end
network_structure(no_of_hidden_layers+1) = no_of_outputs;


weight_matrix = zeros(network_structure(1), no_of_inputs);
bias_matrix = zeros(network_structure(1),1);

% READING THE INPUT WEIGHT MATRIX
for i = 1:network_structure(1)
    for j = 1:no_of_inputs
        weight_matrix(i,j) = file_data(pointer);
        pointer = pointer + 1;
    end
    bias_matrix(i) = file_data(pointer);
    pointer = pointer + 1;
end
nn_controller.weights{1}=weight_matrix;
nn_controller.biases{1}=bias_matrix;
f=cell(size(bias_matrix,1),1);
f(:)={'sigmoid'};
nn_controller.layers{1}=f;


for i = 1:(no_of_hidden_layers)
    
    weight_matrix = zeros(network_structure(i+1), network_structure(i));
    bias_matrix = zeros(network_structure(i+1),1);

    % READING THE WEIGHT MATRIX
    for j = 1:network_structure(i+1)
        for k = 1:network_structure(i)
            weight_matrix(j,k) = file_data(pointer);
            pointer = pointer + 1;
        end
        bias_matrix(j) = file_data(pointer);
        pointer = pointer + 1;
    end
    
    nn_controller.weights{i+1}=weight_matrix;
    nn_controller.biases{i+1}=bias_matrix;
    if i < no_of_hidden_layers
        f=cell(size(bias_matrix,1),1);
        f(:)={'sigmoid'};
        nn_controller.layers{i+1}=f;
    end
end

end