function [dx]=Quad_12(t,x,Control, noise)

dx(1,1) = cos(x(8))*cos(x(9))*x(4) + (sin(x(7))*sin(x(8))*cos(x(9)) - cos(x(7))*sin(x(9)))*x(5) + (cos(x(7))*sin(x(8))*cos(x(9)) + sin(x(7))*sin(x(9)))*x(6)+noise(1);
dx(2,1) = cos(x(8))*sin(x(9))*x(4) + (sin(x(7))*sin(x(8))*sin(x(9)) + cos(x(7))*cos(x(9)))*x(5) + (cos(x(7))*sin(x(8))*sin(x(9)) - sin(x(7))*cos(x(9)))*x(6)+noise(2);
dx(3,1) = sin(x(8))*x(4) - sin(x(7))*cos(x(8))*x(5) - cos(x(7))*cos(x(8))*x(6)+noise(3);

dx(4,1) = x(12)*x(5) - x(11)*x(6) - 9.81*sin(x(8))+noise(4);
dx(5,1) = x(10)*x(6) - x(12)*x(4) + 9.81*cos(x(8))*sin(x(7))+noise(5);
dx(6,1) = x(11)*x(4) - x(10)*x(5) + 9.81*cos(x(8))*cos(x(7)) - 9.81 -  Control(1,1)/1.4 +noise(6);
  
dx(7,1) = x(10) + (sin(x(7))*(sin(x(8))/cos(x(8))))*x(11) + (cos(x(7))*(sin(x(8))/cos(x(8))))*x(12)+noise(7);
dx(8,1) = cos(x(7))*x(11) - sin(x(7))*x(12)+noise(8);
dx(9,1) = (sin(x(7))/cos(x(8)))*x(11) + (cos(x(7))/cos(x(8)))*x(12)+noise(9);
  
dx(10,1) = -0.92592592592593*x(11)*x(12) + 18.51851851851852*Control(2,1)+noise(10);
dx(11,1) = 0.92592592592593*x(10)*x(12) + 18.51851851851852*Control(3,1)+noise(11);
dx(12,1) = noise(12);


end
