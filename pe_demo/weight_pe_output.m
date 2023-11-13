output1 = out.o_sum_out1.DATA(1:10).';
output2 = out.o_sum_out2.DATA(1:10).';
delta = out.delta_sum_out.DATA(1:10).';
oi = out.oi_out.DATA(1:10).';

display(output1);
display(output2);
display(delta);
display(oi);