function shaded_var_color(Data, color)


X= min(Data,[],2);
Y= max(Data,[],2);

t= 0:size(Data,1)-1;
fill([t, fliplr(t)], [X', fliplr(Y')], color, 'FaceAlpha', 1, 'EdgeColor' , 'none');

end

