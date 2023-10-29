output_out = out.output_out.DATA(1:10).';
delta_out = out.delta_out.DATA(1:10).';
new_weight = out.update_weight.DATA(1:10).';

display(output_out);
display(delta_out);
display(new_weight);