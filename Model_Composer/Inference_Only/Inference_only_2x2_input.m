num_bits =16;
frac_bits=8;
num_layer = 2;

% weights & biases: 
w00=[-1.32 1.054]; w01=[2.20 0.61];
w10=[1.29  0];     w11=[-2.25 0];
b1=[-0.054 0.002]; b2=[-0.055 0];

w00=[1 2]; w01=[1 2];
w10=[1 2];     w11=[1 2];
b1=[1 2]; b2=[1 2];

% inputs
o0_in = timeseries([ 1; 1; 1; 1;1;1]);
o1_in = timeseries([ 1; 1; 1; 1;1;1]);

input_control = timeseries([0;0;1;0;1;0;1]);
layer = timeseries([0;1;0;1;0;1;0;1;0;1]); 
backprop_on = timeseries([0]);

MSE_on = timeseries([0; 0; 0; 0; 1; 0; 0]); %need to match when forward prop finishes. 
label_0 = timeseries(zeros(8,1)*1); %consistently outputs 1 for now 
label_1 = timeseries(zeros(8,1)*1);

net_in = timeseries([10.5;0;0;0;0]);
