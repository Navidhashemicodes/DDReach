function  Star_sets = ReLUplex_Reachability_ss(Center, epsilon, Net, analysis_type)


system_dim = size(Center,1);




%%%%%%%%%%%%%%%%%%


init_star_state = Star();    %%% consider the following lines, this how I initiate the Star() set, my experiments shows, if I initiate Star() set in another way, then I will face computational errors.
init_star_state.V = [Center eye(system_dim)];
init_star_state.C = zeros(1,system_dim);
init_star_state.d = 0;
init_star_state.predicate_lb = -epsilon;
init_star_state.predicate_ub = epsilon;
init_star_state.dim = system_dim;
init_star_state.nVar = system_dim;

in{1} = init_star_state;   %%%% This is the Star() set that will be introduced to the neural network. I am thinking to upgrade the in away it accept multiple Star() sets. In that case we can also include non-convex sets in our research.

len = length(Net.layers);
for i = 1:len
    disp(['analysis of Layer ' num2str(i) ' of ' num2str(len) '.'])
    lenS = length(in{i});
    lenAC = 0;
    for k = 1:lenS    %%%% As we know the number of the Star() sets increases over the network, thus we should pass them one by one. This for loop introduces the Star() sets one by one to NNV toolbox.
        in_layer{i}(k) = affineMap(in{i}(k), Net.weights{i}, Net.biases{i});   %%% Maps the star set through weight matri and bias vectors,  in_layer is an star set that is introduced to the layer.
        
        In_layer = in_layer{i}(k);
        if strcmp(Net.layers{i}(1), 'purelin')   %%% Our layers in Trapezium-FFNN consists of linear and nonlinear activation functions. Where the location of linear and nonlinear activations are separated. 
                                                 %%% the If command is finding the location of the first nonlinear activation function in the layer. (index)
            index = 1;
            dd = true;
            while dd
                index = index+1;
                if ~strcmp(Net.layers{i}(index), 'purelin')
                    dd = false;
                end
            end
        else
            index = 1;
        end
        In_layer.V = In_layer.V(index:end, :);   %%% over the next line of codes we separate the dimesions od star set the are affected with relu activations from the dimensions that are not.
        if size(In_layer.state_lb , 1 )~=0     %%%% sometimes set is defined with its upper and lower state bounds , and some other times with lower and upper bounds on the predicate. 
                                               %%%% Here we want to check what condition holds now on the star set and then we separate the linear and nonlinear dimensions accorrdingly.
            In_layer.state_lb = In_layer.state_lb(index:end, 1);
            In_layer.state_ub = In_layer.state_ub(index:end, 1);
        end
        In_layer.dim =  size(In_layer.V , 1 );
        
        
        Layers = LayerS(eye(In_layer.dim), zeros(In_layer.dim,1) , 'poslin');
        Ln     = LayerS(eye(In_layer.dim), zeros(In_layer.dim,1) , 'purelin');
        Layers = [Layers Ln];
        
        F = FFNNS(Layers);
        %%%%    'exact-star'   or    'approx-star'
        [Out_layer, ~] = reach(F,In_layer, analysis_type, []);    %%% The star set is introduced to the layer by calling nnv toolbox. This returns the out put of the layer as a new star set. 
        
        lenAC_now = length(Out_layer);
        for j = 1:lenAC_now    %%% every single (1 out of lenS) star set that is introduced to the layer will result in multiple (lenAC_now) star sets . Thus we collect all of them in this for loop inside in{i+1}. 
            
            if strcmp(analysis_type, 'exact-star')  %%% The inaffected elements are stacked on the affected elements after paassing through the layer.
                Out_layer(j).V = [in_layer{i}(k).V(1:index-1,:)  ;  Out_layer(j).V];   %%% you should check if the predicate updating process affects the V or not, I guess No, but you need to check
            elseif strcmp(analysis_type, 'approx-star') %%% After passing the Affected elements through relu layers with approx-star technique. The number of their predicates can increase thus we stack the predicates 
                                                        %%% of inaffected elements and affected elements like this
                d_size = size(Out_layer(j).V,2)-size(in_layer{i}(k).V(1:index-1,:), 2);
                Out_layer(j).V = [[in_layer{i}(k).V(1:index-1,:) zeros(index-1, d_size) ] ;  Out_layer(j).V];
            end
            
            if size(in_layer{i}(k).state_lb , 1 )~=0
                if size(Out_layer(j).state_lb , 1 )~=0 %%%% if before and after passing the star through the relu layer, the star contains bounds on the states, then the bounds will be stacked like this 
                    Out_layer(j).state_lb = [in_layer{i}(k).state_lb(1:index-1,1) ; Out_layer(j).state_lb];
                    Out_layer(j).state_ub = [in_layer{i}(k).state_ub(1:index-1,1) ; Out_layer(j).state_ub];
                end
            else
                if size(Out_layer(j).state_lb , 1 )~=0 %%% if we didnt have bounds on states before passing the star. but have it right now, then the bounds will be stacked like this.
                    SSS =  getBox(in_layer{i}(k));
                    Out_layer(j).state_lb = [SSS.lb(1:index-1,1) ; Out_layer(j).state_lb];
                    Out_layer(j).state_ub = [SSS.ub(1:index-1,1) ; Out_layer(j).state_ub];
                end
            end

            Out_layer(j).dim = Out_layer(j).dim + (index-1);   %%% index-1 is the number of linear activations while Out_layer(j).dim is the number of elements affected by nonlinear activations, or 
                                                               %%% in another word the number of nonlinear activations
            in{i+1}(lenAC+j) = Out_layer(j);
        end
        lenAC     = lenAC+ lenAC_now;
    end 
end


lenS = length(in{len+1});
for k = 1:lenS   %%% The following for loop takes the set of stars produced at the end of the process. and computes the lower bound of their union. This is nothing but the robustness range for STL specification.
    Star_sets(k) = affineMap(in{len+1}(k), Net.weights{end}, Net.biases{end});
end

end



