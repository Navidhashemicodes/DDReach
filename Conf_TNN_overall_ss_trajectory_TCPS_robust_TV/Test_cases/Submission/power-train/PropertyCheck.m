% Powertrain Control Verification Benchmark Experimental File
% Created by Xiaoqing Jin (xiaoqing.jin@tema.toyota.com)
% Powertrain Control Model Based Development
% Toyota Technical Center 
% Jan 27th 2014
clc

disp('please select the interested requirement:')
disp('(1) Closed-loop transient requirement (formula 26)')
disp('(2) Closed-loop settling time requirement (formula 27)')
disp('(3) Error tolerrance (formula 29)')
disp('(4) Rich case excursion (formula 30)')
disp('(5) Lean case excursion (formula 31)')
disp('(6) Transition out of power mode (formula 32)')
disp('(7) Power mode performance (formula 33)')
disp('(8) Startup and sensor_fail modes performance (formula 34)')
disp(' ')
user_input = input('Select a requirement (1-8):')
%%
%%
% Initialize the model
open AbstractFuelControl_M2;
model = @BlackBoxAbstractFuelControl;
c1=0.41328
c2=-0.366
c3=0.08979
c4=-0.0337
c5=0.0001
c6=2.821
c7=-0.05231
c8=0.10299
c9=-0.00063
c10=1.0
c11=14.7
c12=0.9
c13=0.04
c14=0.14
% total simulation time
simTime = 50 ; 
% time to start measurement, mainly used to ignore 
% Simulink initialization phase
measureTime = 1;  
% number of control points, here we use onstant engine speed
cp_array = 1 ; 
% engine speed range
input_range = [ 900  1100]; 
% time to introduce sensor failure (default sensor failure won't happen)
fault_time = 100; 
% setting time
eta = 1;
% parameter h used for event definition
h = 0.02;
% parameter related to the period of the pulse signal
zeta_min = 5;
%
C = 0.05;
Cr = 0.1;
Cl = 0.1;
Ut = 0.05;
%
taus = 10 + eta;


% default settings
spec_num = 1; %specification measurement
c23=1.0;
c24=1.0;
c25=1.0;
initial_cond = [0 61.1;10 30];

switch user_input
    case 1 % transinet requirement of close-loop pulse response (formula 26) 
        Ut = 0.05;
        phi = ['[]_(' num2str(taus) ', ' num2str(simTime) ') utl /\ utr'];
    case 2 % close-loop pulse response (formula 27)
        Ut = 0.02;
        phi = ['[]_(' num2str(taus) ', ' num2str(simTime) ')(((low/\<>_(0,' ...
            num2str(h) ')high) \/ (high/\<>_(0,' num2str(h) ')low))' ...
            '-> []_[' num2str(eta) ', ' num2str(zeta_min) '](utr /\ utl))'];
        
    case 3  % closed loop pulse error tolerance (formula 29)
        spec_num = 2;
        Ut = C;
        phi = ['<>_['  num2str(simTime)  ','  num2str(simTime) '] utr' ];
       
    case 4  % Rich case excursion (formula 30)
        fuel_inj_tol = 1.05;
        MAF_sensor_tol = 1.05;
        AF_sensor_tol = 1.01;
        Ut = Cr;
        phi = ['[]_(' num2str(taus) ', ' num2str(simTime) ')utr'];
        
    case 5 % Lean case excursion (formula 31)
        fuel_inj_tol = 0.95;
        MAF_sensor_tol = 0.95;
        AF_sensor_tol = 0.99;   
        Ut = Cl;
        phi = ['[]_(' num2str(taus) ', ' num2str(simTime) ')utl'];
        
    case 6 % Transition out of power mode (formula 32)
        initial_cond = [61.3 81.2;5 10];
        Ut = 0.02;
        phi = ['[] ((pwr /\ <>_(0,' num2str(h), ')norm) -> ([]_(' ...
            num2str(eta) ', ' num2str(zeta_min) ') utl /\ utr))'];
               
    case 7 % Power mode performance (formula 33)
        initial_cond = [61.3 81.2;10 20];
        Ut = 0.2;
        phi = '[] (pwr -> (utl /\ utr))';
       
    case 8 % Startup and sensor_fail modes performance (formula 34)
        fault_time = 15;
        initial_cond = [0 61.1;10 20];
        Ut = 0.1;
        phi = ['[] (((low/\<>_(0,' num2str(h) ...
            ')high) \/ (high/\<>_(0,' num2str(h) ')low)) -> [] (utr /\ utl))'];
    otherwise
        disp('Wrong measurement option')
end

% definition of predicates
i = 1;
preds(i).str = 'low'; % for the pedal input signal
preds(i).A =  [0 0 1] ;
preds(i).b =  0.5 ;
i = i+1;

preds(i).str = 'high'; % for the pedal input signal
preds(i).A =  [0 0 -1] ;
preds(i).b =  -0.5 ;
i = i+1;

% rise event is represented as low/\<>_(0,h)high
% fall event is represented as high/\<>_(0,h)low

preds(i).str = 'norm'; % mode < 0.5 (normal mode = 0)
preds(i).A =  [0 1 0] ;  
preds(i).b =  0.5 ;
i = i+1;

preds(i).str = 'pwr'; % mode >0.5 (power mode = 1)
preds(i).A =  [0 -1 0] ;
preds(i).b =  -0.5 ;
i = i+1;

preds(i).str = 'utr'; % u<=Ut
preds(i).A =  [1 0 0] ;
preds(i).b =  Ut ;
i = i+1;

preds(i).str = 'utl'; % u>=-Ut
preds(i).A =  [-1 0 0] ;
preds(i).b =  Ut ;
i = i+1;
%%
% options
opt = staliro_options();
%opt.parameterEstimation = 1;
opt.black_box = 1;
opt.SampTime = 0.05;
opt.optimization_solver =  'SA_Taliro'
opt.spec_space = 'Y';
opt.interpolationtype={'const'};
opt.runs = 1;
opt.seed = 2014;  % for repetability
opt.n_workers = 1;
opt.sa_params.n_tests = 1000; % number of tests

disp('Running S-TaLiRo with chosen solver ...')

profile on
tic
[results,history] = staliro(model,initial_cond,input_range,cp_array,phi,preds,simTime,opt);
runtime=toc;
profile off
profile viewer