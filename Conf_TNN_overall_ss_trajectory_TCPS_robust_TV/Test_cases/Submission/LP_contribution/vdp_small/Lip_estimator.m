function L = Lip_estimator(Input,Output)

num=size(Input,2);
N=1000000;
LL=zeros(1,N);
for iii=1:N
    i1=floor(rand*num)+1;
    i2=floor(rand*num)+1;
    LL(iii)=norm(Output(:,i1)-Output(:,i2) , 2)/norm(Input(:,i1)-Input(:,i2) , 2);
end
L=max(LL);
end


