parfor i=1:300000
    noise = mvnrnd(zeros(12,1), 0.0001*eye(12),1)';
    Input_Data2(:,i) = Input_Data2(:,i) + noise;
end
disp('2')
for k=1:300
    Trajec = Output_Data2(13:end,(k-1)*1000+1:k*1000);
    parfor i=1:1000
        Traj = Trajec(:,i)
        for j=1:50
            Traj((j-1)*12+1:j*12,1) = Traj((j-1)*12+1:j*12,1) + mvnrnd(zeros(12,1), 0.0001*eye(12),1)';
        end
        Trajec(:,i) = Traj;
    end
    Output_Data2(13:end,(k-1)*1000+1:k*1000) = Trajec;
end
Output_Data2(1:12,:) = Input_Data2;