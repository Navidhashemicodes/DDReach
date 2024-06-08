function x_next = nonlinear(x, avg, cov)
x_next = zeros(2,1);
eta = mvnrnd(avg, cov, 1)';
x_next(1) = 0.985*x(2)+sin(0.5*x(1))-0.6*sin(x(1)+x(2))-0.07+eta(1);
x_next(2) = 0.985*x(1)+cos(0.5*x(2))-0.6*cos(x(1)+x(2))-0.07+eta(2);
end
