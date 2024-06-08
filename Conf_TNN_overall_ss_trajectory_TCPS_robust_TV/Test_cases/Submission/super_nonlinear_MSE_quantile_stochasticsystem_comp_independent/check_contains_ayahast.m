function  decision = check_contains_ayahast(Star_sets , traj )

i=1;
dd = true;
while dd
    
    SS = Star_sets(1);
    SS = SS{1};
        
    bool = contains(SS(i) , traj);

    if bool == 1
        decision = 1;
        dd = false;
    else
        i = i+1;
    end

    if i > length(SS)
        dd = false;
        decision = 0;
    end


end

end