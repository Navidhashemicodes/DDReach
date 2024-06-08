clear all
close all
clc

c1=0.41328;
c2=-0.366;
c3=0.08979;
c4=-0.0337;
c5=0.0001;
c6=2.821;
c7=-0.05231;
c8=0.10299;
c9=-0.00063;
c10=1.0;
c11=14.7;
c12=0.9;
c13=0.04;
c14=0.14;
% total simulation time
simTime = 20 ; 
% time to start measurement, mainly used to ignore 
% Simulink initialization phase
measureTime = 1;  
fault_time = 100; 
en_speed = 1000;

theta_init = 8.8;
p_init = 0.725; % +/- 10%
% p_init = ( 0.9 + rand*0.2 )*p_init;
lambda_init = 14.7; % +/- 1%
% lambda_init = ( 0.9 + rand*0.2 )*lambda_init;
p_est_init = 0.5455; % 
% p_est_init = ( 0.9 + rand*0.2 )*p_est_init;
i_init = 0; % [0 0.1]
% i_init = 0.1*rand;
mode_init = 1;



c23=0.9+0.2*rand; %% C23 is sensor 
c24=0.9+0.2*rand;
c25=0.9+0.2*rand;

sim("AbstractFuelControl_M2_Navid.slx")
% % pick time horizon of 20, only track reach states after time 10
% % 0.02, 0.05, 0.1 (objective: reduce conservatism) 
dt = 0.1;
t = 0:dt:simTime;
t_len = length(t);
T = ScopeData.time;
T_len = length(T); 
S = zeros(t_len,6);

S(:,1) = interp1(T , ScopeData.signals(1).values , t');
S(:,2) = interp1(T,  ScopeData.signals(2).values , t');
S(:,3) = interp1(T,  ScopeData.signals(3).values , t');
S(:,4) = interp1(T,  ScopeData.signals(4).values , t');
S(:,5) = interp1(T,  ScopeData.signals(5).values , t');
S(:,6) = interp1(T,  ScopeData.signals(6).values , t');









