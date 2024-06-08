function  Box = Overall_Box( Star_sets, weight, bias)



lenS = length(Star_sets);
for k = 1:lenS 
    range(k) = affineMap(Star_sets(k), weight, bias);
    
    if size(range(k).state_lb , 1 ) ==0
        SSS = getBox(range(k));
        the_lb(:,k) = SSS.lb;
        the_ub(:,k) = SSS.ub;
    else
        the_lb(:,k) = range(k).state_lb;
        the_ub(:,k) = range(k).state_ub;
    end
end

if size(the_lb,2)>1
    Lb = min(the_lb')';
    Ub = max(the_ub')';
else
    Lb = the_lb;
    Ub = the_ub;
end

Box = [ Lb , Ub ];



end