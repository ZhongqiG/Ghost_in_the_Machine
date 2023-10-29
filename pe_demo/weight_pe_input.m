num_bits = 12;
rst_in = [1; zeros(6,1)];
eta_in = [0.5; zeros(6,1)];
weight_in = [2; zeros(6,1)];
output_in = [0; 0; 1; 3; 5; 7];
delta_in = [0; 5; 2; 6; 1; 2];
sum_s_in = [1; 5; 2; 3; 9; 8];
sum_o_in = [0; 2; 4; 6; 7; 4];

rst = timeseries(rst_in);
eta = timeseries(eta_in);
weight = timeseries(weight_in);
output = timeseries(output_in);
delta = timeseries(delta_in);
sum_s = timeseries(sum_s_in);
sum_o = timeseries(sum_o_in);