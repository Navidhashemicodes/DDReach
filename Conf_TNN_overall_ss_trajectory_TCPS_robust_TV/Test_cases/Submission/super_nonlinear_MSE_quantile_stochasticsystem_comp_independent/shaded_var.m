function shaded_var(Data)

centers = mean(Data, 2);
X= min(Data,[],2);
Y= max(Data,[],2);

t= 0:size(Data,1)-1;
% plot(t, centers, 'blue');
% hold on
fill([t, fliplr(t)], [X', fliplr(Y')], 'g', 'FaceAlpha', 0.2, 'EdgeColor' , 'none');

end

