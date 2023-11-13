num_bits = 12;
rst_in = [1; zeros(6,1)];
eta_in = [0.5; zeros(6,1)];
init_bias_in = [0; 1; zeros(5,1)];
update = [0; 1; 1; 0; 0; 1; 0];
o_sum = [0; 0; 3; 2; 1; 2; 3];
delta = [0; 0; 5; 2; 6; 1; 2];

rst = timeseries(rst_in);
eta = timeseries(eta_in);
init_bias = timeseries(init_bias_in);
update_in = timeseries(update);
o_sum_in = timeseries(o_sum);
delta_in = timeseries(delta);