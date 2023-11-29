num_bits =8;
% Test case 1: 
rst_in = [1; zeros(6,1)];
eta_in = [0.5; zeros(6,1)];
alpha_in = [0.1; zeros(6,1)];
init_weight00_in = [0.25; 0; 0; 0; 0; 0; 0];
init_weight01_in = [0.5; 0; 0; 0; 0; 0; 0];
update_in = [0; 1; 0; 0; 0; 0; 0; 0];
init_bias_in = [0; 0; 1.5; 0; 0; 0; 0];
o_sum_in = [0; 0; 1.5; 0; 0; 0; 0];
delta2_in = [0; 0; 3; 0; 0; 0; 0];
delta00_in = [0; 0; 0; 0; 0; 0; 0];
delta01_in = [0; 0; 0; 0; 0; 0; 0];
oi00_in = [0; 0; 2; 0; 0; 0; 0];
oi01_in = [0; 0; 0; 3; 0; 0; 0];



