num_bits = 12;
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

rst = timeseries(rst_in);
eta = timeseries(eta_in);
alpha = timeseries(alpha_in);
init_weight00 = timeseries(init_weight00_in);
init_weight01 = timeseries(init_weight01_in);
update = timeseries(update_in);
init_bias = timeseries(init_bias_in);
o_sum = timeseries(o_sum_in);
delta2 = timeseries(delta2_in);
delta00 = timeseries(delta00_in);
delta01 = timeseries(delta01_in);
oi00 = timeseries(oi00_in);
oi01 = timeseries(oi01_in);

