num_bits = 12;
% First cycle is reset; sent eta in first cycle. 
% In second cycle, send initial weight in; remember to set update to high.
% Wait till third cycle to input first set of values
% The weight register should keep the weight value until you set update to
% high. 
rst_in = [1; zeros(6,1)];
eta_in = [0.5; zeros(6,1)];
init_weight_in = [0; 1; zeros(5,1)];
o_sum1 = [0; 0; 3; 2; 1; 2; 3];
o_sum2 = [0; 0; 3; 2; 1; 2; 3];
update = [0; 1; 1; 0; 0; 1; 0];
oi = [0; 0; 2; 4; 6; 7; 4];
delta = [0; 0; 5; 2; 6; 1; 2];

rst = timeseries(rst_in);
eta = timeseries(eta_in);
init_weight = timeseries(init_weight_in);
o_sum_in1 = timeseries(o_sum1);
o_sum_in2 = timeseries(o_sum2);
update_in = timeseries(update);
oi_in = timeseries(oi);
delta_in = timeseries(delta);