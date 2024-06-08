function  decision = check_contains_agenist(Star_sets , traj , n )



horizon_with_zero = length(traj)/n;

j=1;  %%% dont start it from 0 , you dont need to check the initial state
ddd = true;

while ddd

    j=j+1;
    

    if j>horizon_with_zero
        decision = 1;
        break;
    end
    
    dd = true;
    i=1;
    while dd

        SS = Star_sets(1);
        SS = SS{1};

        SSS = SS(i);
        SSS.V = SS(i).V( (j-1)*n+1 : j*n, : );
        SSS.dim = n;
        
        if ~isempty(SS(i).state_lb)
            SSS.state_lb = SS(i).state_lb( (j-1)*n+1 : j*n, : );
            SSS.state_ub = SS(i).state_ub( (j-1)*n+1 : j*n, : );
        end

        if ~isempty(SS(i).Z)
            error('I dont know what is happening here!!!')
        end

        bool = contains( SSS , traj( (j-1)*n+1 : j*n, : ) );

        if bool == 1
            dd = false;
        else
            i = i+1;
        end

        if i > length(SS)
            dd = false;
            ddd = false;
            decision = 0;
        end


    end


end

end