function dx = vannderpol(t,x, noise)

mu = -1;

dx(1,1) = x(2) + noise(1);
dx(2,1) = mu*(1-x(1)^2)*x(2) - x(1) + noise(2);

end
