% PE Array
% This function has to come first according to Matlab convention
% All other functions in this file are used in this main array function

function [output_k, delta_kmin1, weight_changes, bias_changes] = pe_array(weights, biases, output_kmin1, delta_k, eta, model, alpha)
    % inputs: 
    % weights - 2x2 array holding all the weights for the PEs
    % biases - 2x1 array holding the biases for the two neurons
    % output_kmin1 - previous layer output array for each neuron (2x1 array
    % where the first value is the output of the first neuron from the
    % previous layer and the second value is the output from the second
    % neuron in the first layer)
    % delta_k - 2x1 array holding the backpropagation from the previous
    % layer
    % eta - weight used to calculate the new weights
    % model - name of the activation function used (currently only
    % implemented for Sigmoid) as a string
    % alpha - weight used to adjust ReLU into ELU

    % outputs:
    % output_k - 2x1 array of the output from each neuron in the layer
    % delta_kmin1 - 2x1 array of the backpropagation
    % weight_changes - the new 2x2 array of weights for the weight PEs
    % bias_changes - the new 2x1 array of biases for the neurons

    % NOTE: This function works on a cycle-by-cyle basis. That is, every
    % input is either a scalar or an array where each value goes to a
    % specific PE. For a time sequence of inputs, this function (currently)
    % must be called with all inputs specified for each clock cycle. Future
    % implementation may calculate the output for the full time sequence on
    % one call.
    
    % define the initial weights and biases for ease of reading
    init_weight00 = weights(1, 1);
    init_weight01 = weights(1, 2);
    init_weight10 = weights(2, 1);
    init_weight11 = weights(2, 2);
    
    init_bias1 = biases(1);
    init_bias2 = biases(2);
    
    % initialize output arrays
    output_k = zeros(2, 1);
    delta_kmin1 = zeros(2, 1);
    weight_changes = zeros(2, 2);
    bias_changes = zeros(2, 1);
    
    % first weight PE, first neuron
    [sum_delta_out00, sum_output_out00, weight_changes(1, 1)] = weights_pe(delta_k(1), output_kmin1(1), init_bias1, 0, init_weight00, eta);
    
    % second weight PE, first neuron
    [sum_delta_out01, net_sum0, weight_changes(1, 2)] = weights_pe(delta_k(1), output_kmin1(2), sum_output_out00, 0, init_weight01, eta);
    
    % output of the first neuron
    output_k(1) = act_pe(net_sum0, model, alpha);
    
    %----------------------------------------------------------------------
    
    % first weight PE, second neuron
    [sum_delta_out10, sum_output_out10, weight_changes(2, 1)] = weights_pe(delta_k(2), output_kmin1(1), init_bias2, sum_delta_out00, init_weight10, eta);
    
    % second weight PE, second neuron
    [sum_delta_out11, net_sum1, weight_changes(2, 2)] = weights_pe(delta_k(2), output_kmin1(2), sum_output_out10, sum_delta_out01, init_weight11, eta);
    
    % output of the second neuron
    output_k(2) = act_pe(net_sum1, model, alpha);
    
    % update the backpropagation outputs (from second to first layer)
    delta_kmin1(1) = error_pe(output_kmin1(1), sum_delta_out10, model);
    delta_kmin1(2) = error_pe(output_kmin1(2), sum_delta_out11, model);
    
    % update to the bias of neuron 1
    k1 = delta_k(1);
    bias_changes(1) = k1*eta - init_bias1;
    
    % update to the bias of neuron 2
    k2 = delta_k(2);
    bias_changes(2) = k2*eta - init_bias2;
    
end

% Processing Elements

% Updates to the weights

function [sum_delta_out, sum_output_out, weight_change] = weights_pe(delta_k, output_kmin1, partial_sum_out_k, partial_sum_delta_k, init_weight, eta)
    % seq_len = size(delta_k);
    % sum_delta_out = zeros(seq_len, 1);
    % sum_output_out = zeros(seq_len, 1);
    % weight_change = zeros(seq_len, 1);
    % weight = init_weight;
    % for i = 1:seq_len
    %     delta = delta_k(i);
    %     output = output_kmin1(i);
    %     sum_o = partial_sum_out_k(i);
    %     sum_s = partial_sum_delta_k(i);
    %     sum_delta_out(i) = sum_s + delta*weight;
    %     sum_output_out(i) = sum_o + output*weight;
    %     weight = output*delta*eta - weight;
    %     weight_change(i) = weight;
    % end
    % the above was commented out so that there was not an issue with array
    % lengths.
    % inputs:
    % delta_k - backprop signal
    % output_kmin1 - previous layer output signal
    % partial_sum_out_k - partial output sum carry in
    % partial_sum_delta_k - partial backprop sum carry in
    % init_weight - weight for the current clock cycle, will get updated as
    % well
    % eta - weight multiplier used in the PE weight update
    % outputs:
    % sum_delta_out - partial backprop sum carry out
    % sum_output_out - partial output sum carry out
    % weight_change - new PE weight for the next clock cycle
    weight = init_weight;
    delta = delta_k;
    output = output_kmin1;
    sum_o = partial_sum_out_k;
    sum_s = partial_sum_delta_k;
    sum_delta_out = sum_s + delta*weight;
    sum_output_out = sum_o + output*weight;
    weight = output*delta*eta - weight;
    weight_change = weight;
end

% Updates to the biases - THIS MAY BE UNCOMMENTED AND ADDED BACK INTO ARRAY

% function [net_sum, bias_change] = bias_pe(delta_k, sum_in, init_bias, eta)
%     seq_len = size(delta_k);
%     bias = init_bias;
%     net_sum = zeros(seq_len, 1);
%     bias_change = zeros(seq_len, 1);
%     for i = 1:seq_len
%         sum = sum_in(i);
%         k = delta_k(i);
%         net_sum(i) = bias + sum;
%         bias = k*eta - bias;
%         bias_change(i) = bias;
%     end
% end

% Applying the Activation Function (ReLU, ELU, and Sigmoid)
function output = act_pe(net_in, model, alpha)
    % seq_len = size(net_in);
    % output = zeros(seq_len, 1);
    % for i = 1:seq_len
    %     omega = net_in(i);
    %     if model == "sigmoid"
    %         output(i) = 1 / (1 + exp(-omega));
    %     elseif model == "relu"
    %         output(i) = max(0, omega);
    %     elseif model == "elu"
    %         if omega >= 0
    %             output(i) = omega;
    %         else
    %             output(i) = alpha * (exp(omega)-1);
    %         end
    %     else
    %         print("error")
    %     end
    % end
    % the above was commented out so that there was not an issue with array
    % lengths.
    % inputs:
    % net_in - complete partial sum of the weight PE outputs
    % model - name of the activation function, as a string
    % alpha - weight used in the construction of ELU
    % outputs:
    % output - output of the neuron
    omega = net_in;
    if model == "sigmoid"
        output = 1 / (1 + exp(-omega));
    elseif model == "relu"
        output = max(0, omega);
    elseif model == "elu"
        if omega >= 0
            output = omega;
        else
            output = alpha * (exp(omega)-1);
        end
    else
        print("error")
    end
end

% Error Propagation Processing Element
function delta_kmin1 = error_pe(output_kmin1, partial_sum_delta_k, model)
    % this processing element produces the derivative of f(net)
    % for example, we can calculate this for f = sigmoid fn
    % this is simply d = o(1-o)delta

    % delta_kmin1 = output_kmin1 * (1 - output_kmin1) * partial_sum_delta_k;

    % however, we are not always using sigmoid. this means we need to
    % actually calculate the derivative of the activation function.

    % inputs:
    % output_kmin1 - 2x1 array representing the output from the previous
    % layer
    % partial_sum_delta_k - running sum for the backpropagation
    % model - name of the activation function as a stringt

    if model == "sigmoid"
        delta_kmin1 = output_kmin1 * (1 - output_kmin1) * partial_sum_delta_k;
    elseif model == "relu"
        delta_kmin1 = 0; % need to implement these!!!
    elseif model == "elu"
        delta_kmin1 = 0;
    else
        print("Error")
    end
end

function final_layer_delta = final_layer_error_pe(output, target, cost)
    % output - output from the last layer of calculation
    % target - target vector for training
    % cost - name of the cost function used
    % note that this final error layer is only implemented for the sigmoid
    % activitation function being used throughout the network
    if cost == "mean_squared_error"
        final_layer_delta = output_kmin1 - target;
    elseif cost == "exponentially_weighted"
        % to be implemented
    end
end



