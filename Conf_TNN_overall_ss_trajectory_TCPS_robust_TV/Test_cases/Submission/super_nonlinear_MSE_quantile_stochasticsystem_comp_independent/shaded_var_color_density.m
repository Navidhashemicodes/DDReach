function shaded_var_color_density(Data, starting, ending, numbins)


X= min(Data,[],2);
Y= max(Data,[],2);

num_data = size(Data, 2);
horizon  = size(Data, 1);

t = 0:size(Data,1)-1;

hold on;
plot(t+1, Y', 'k', 'LineWidth', 0.2);
plot(t+1, X', 'k', 'LineWidth', 0.2);


% cmap = zeros(numbins, 3);
% for i = 1: numbins
%     cmap(i,:) = starting + (ending -starting) * ( (i-1)/(numbins-1) ) ;
% end
% 
% 
% for i = 1:horizon-1
% 
%     binedges = linspace(X(i), Y(i) , numbins+1);
%     Xi = sort(Data(i,:));
%     counts = zeros(1, numbins);
%     k  = 2;
%     counts(k-1) = 0;
%     for j=1:num_data
% 
%         if Xi(j) < binedges(k) 
%             counts(k-1) = counts(k-1) + 1;
%         else
%             k = k+1;
%         end
%     end
%     Normalized_counts = counts/max(counts);
% 
%     for j = 1 : numbins
%         color = cmap( floor( Normalized_counts(j) * (numbins-1) ) + 1   , : );
%         xvertices = [i i+1 i+1 i]-0.5;
%         yvertices = [binedges(j) binedges(j) binedges(j+1) binedges(j+1)];
%         fill(xvertices , yvertices , color , 'FaceAlpha', 0.3, 'EdgeColor' , 'none')
%     end
% 
%     colormap(cmap);
%     colorbar;

end

