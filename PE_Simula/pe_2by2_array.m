% converting Python code into Matlab for future development

%% PE Array
% This function has to come first according to Matlab convention
% All other functions in this file are used in this main array function

function [output_k, delta_kmin1, weight_changes, bias_changes] = pe_array(weights, biases, output_kmin1, partial_sum_delta_k, delta_k, partial_sum_out_k, eta, model, alpha)
    % inputs: 
    % weights - array representing all the weights for the 2x2 array
    % output_kmin1 - previous layer output array for each neuron
    % partial_sum_delta_k - partial delta sum for current layer from
    % previous neuron in the same layer
    % delta_k - propagate delta for current layer, current neuron
    % partial_sum_out_k = partial sum output for current layer, same neuron
    
    % define the initial weights
    init_weight11 = weights(1, 1);
    init_weight12 = weights(1, 2);
    init_weight21 = weights(2, 1);
    init_weight22 = weights(2, 2);
    
    init_bias1 = biases(1);
    init_bias2 = biases(2);
    
    % initialize output arrays
    output_k = zeros(2, 1);
    delta_kmin1 = zeros(2, 1);
    weight_changes = zeros(2, 2);
    bias_changes = zeros(2, 1);
    
    % first weight PE, first neuron
    sum_delta_out11, sum_output_out11, weight_changes(1, 1) = weights_pe(delta_k(1), output_kmin1(1), partial_sum_out_k(1), partial_sum_delta_k(1), init_weight11, eta);
    
    % second weight PE, first neuron
    sum_delta_out12, sum_output_out12, weight_changes(1, 2) = weights_pe(delta_k(1), output_kmin1(2), sum_output_out11, partial_sum_delta_k(2), init_weight12, eta);
    
    % output of the first neuron
    net_sum1, bias_changes(1) = bias_pe(delta_k(1), sum_output_out12, init_bias1, eta);
    output_k(1) = act_pe(net_sum1, model, alpha);
    
    %----------------------------------------------------------------------
    
    % first weight PE, second neuron
    sum_delta_out21, sum_output_out21, weight_changes(2, 1) = weights_pe(delta_k(2), output_kmin1(1), partial_sum_out_k(1), sum_delta_out11, init_weight21, eta);
    
    % second weight PE, second neuron
    sum_delta_out22, sum_output_out22, weight_changes(2, 2) = weights_pe(delta_k(2), output_kmin1(2), sum_output_out21, sum_delta_out12, init_weight22, eta);
    
    % output of the second neuron
    net_sum2, bias_changes(2) = bias_pe(delta_k(2), sum_output_out22, init_bias2, eta);
    output_k(2) = act_pe(net_sum2, model, alpha);
    
    % update the backpropagation outputs
    delta_kmin1(1) = sum_delta_out21;
    delta_kmin1(2) = sum_delta_out22;
    
end
%% Processing Element Components

% Updates to the weights

function [sum_delta_out, sum_output_out, weight_change] = weights_pe(delta_k, output_kmin1, partial_sum_out_k, partial_sum_delta_k, init_weight, eta)
    seq_len = size(delta_k);
    sum_delta_out = zeros(seq_len, 1);
    sum_output_out = zeros(seq_len, 1);
    weight_change = zeros(seq_len, 1);
    weight = init_weight;
    for i = 1:seq_len
        delta = delta_k(i);
        output = output_kmin1(i);
        sum_o = partial_sum_out_k(i);
        sum_s = partial_sum_delta_k(i);
        sum_delta_out(i) = sum_s + delta*weight;
        sum_output_out(i) = sum_o + output*weight;
        weight = output*delta*eta - weight;
        weight_change(i) = weight;
    end
end

% Updates to the biases

function [net_sum, bias_change] = bias_pe(delta_k, sum_in, init_bias, eta)
    seq_len = size(delta_k);
    bias = init_bias;
    net_sum = zeros(seq_len, 1);
    bias_change = zeros(seq_len, 1);
    for i = 1:seq_len
        sum = sum_in(i);
        k = delta_k(i);
        net_sum(i) = bias + sum;
        bias = k*eta - bias;
        bias_change(i) = bias;
    end
end

% Applying the Activation Function (ReLU, ELU, and Sigmoid)
function output = act_pe(net_in, model, alpha)
    seq_len = size(net_in);
    output = zeros(seq_len, 1);
    for i = 1:seq_len
        omega = net_in(i);
        if model == "sigmoid"
            output(i) = 1 / (1 + exp(-omega));
        elseif model == "relu"
            output(i) = max(0, omega);
        elseif model == "elu"
            if omega >= 0
                output(i) = omega;
            else
                output(i) = alpha * (exp(omega)-1);
            end
        else
            print("error")
        end
    end
end