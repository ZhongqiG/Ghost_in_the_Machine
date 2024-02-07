'''
Maya Borowicz, using code developed by Davis Jackson and James Ding
02/05/2024

This Class can be used to simulate the training and testing
of data objects using the GIM architecture.
'''

# Import packages
import numpy as np

class GIM_simulation:
    def __init__(self, activation_function="relu", weight_array=[], bias_array=[], alpha=0.1, learning_rate=0.1):
        # Initialize the data object

        self.weights = []
        self.biases = []
        self.layers = 0

        self.set_initial_condition(weight_array, bias_array)
        
        self.activation_function = activation_function

        self.alpha = alpha
        self.learning_rate = learning_rate

    def set_initial_condition(self, initial_weights_array, initial_biases_array):
        # Add an array with the initial weights and baises

        # Check that the layers have the same nodes
        if not self.check_parameter_dimensions():
            print("ERROR: Dimensions do not match between weights and biases")
        else:
            self.weights = initial_weights_array
            self.biases = initial_biases_array
            self.layers = len(self.weights)
        
    def set_random_initial_condition(self, layer_array, number_times_generated=1):
        # Set random weights and biases based off the nodes and layers

        # Set a random seed
        np.random.seed(2)
        
        # Create containers for the generated random weights and biases 

        # Based on the number of times weights should be generated
        for i in range(number_times_generated):
            initial_weights = []
            initial_biases = []

            # Create random weights and biases for each layer
            for idx in range(len(layer_array) - 1):
                initial_weights.append(np.random.rand(layer_array[idx + 1], layer_array[idx]))

            for idx in range(len(layer_array) - 1):
                initial_biases.append(np.random.rand(layer_array[idx + 1], 1))

        # Set the weights to random weights
        self.weights = initial_weights
        self.biases = initial_biases

        # Set the layers
        self.layers = len(layer_array) - 1 

    def set_activation_function(self, activation_function):
        # Change activation function
        self.activation_function = activation_function

    def set_alpha(self, alpha):
        # Change the alpha value
        self.alpha = alpha

    def set_learning_rate(self, learning_rate):
        # Change learning rate
        self.learning_rate = learning_rate

    def test(self, data_points):
        # Runs the data points through the weights and biases
        # returns the computed outputs

        # Check that weight and bias arrays are not empty 
        if (self.weights == []) or (self.biases == []):
            print("ERROR: No given weights or biases")
            return None

        # Create a list to store the outputs
        computed_outputs = []

        # Compute the output for each data point
        for data_point in data_points:
            
            # Set the initial input to the layer
            layer_input = data_point

            # Run the data point through each layer
            for idx in range(self.layers):

                # Compute the layer output
                layer_output = np.dot(self.weights[idx], layer_input) + self.biases[idx]

                # Perform the activation function
                # Vectorize the activation function
                vectorized_actiavtion_function = np.vectorize(self.__activation_pe)
                post_activation_output =vectorized_actiavtion_function(layer_output)
                layer_input = post_activation_output

            # Save the output for this data point
            computed_outputs.append(post_activation_output)

        return computed_outputs

    def train(self, input_data_points, expected_outputs, num_iteration):
        # Trains the inputted training data based on set parameters

        # Create an archive of the mean squared error every epoch
        mse_archive = []

        # Create a dictionary to store if the prediction is correct
        prediction_accurate_for_epoch = {}
        for data_point in input_data_points:
            prediction_accurate_for_epoch[str(data_point)] = []

        # Save the number of predictions that are accurate
        number_of_accurate_predictions_per_epoch = []

        # Running of Training Loop
        for epoch in range(num_iteration):

            # Create a variable to store the total squared error for an epoch
            total_squared_error = 0

            # Create variable to store the number of correct predictions in this epoch
            num_correct_predictions = 0

            # Run each piece of training data through the loop
            for idx in range(len(input_data_points)):

                # Extract the inputted data and expected output from the full set of training data
                input_data = input_data_points[idx]
                expected_output = expected_outputs[idx]

                # Create a list to save outputs for each layer
                output_archive = []
                output_archive.append(input_data)

                next_layer_input = input_data

                # Run each input data through the neural network
                for layer_weights, layer_biases in zip(self.weights, self.biases):

                    # Initialize delta to an array of zeros
                    delta = np.zeros((layer_weights.shape[0],1))

                    # Calculate the output after running the data through the layer
                    # Weights and biases are unchanged here
                    [output, dummy_delta, dummy_weights, dummy_biases] = self.__array(layer_weights, layer_biases, next_layer_input, delta, self.learning_rate, self.activation_function, self.alpha, layer_weights.shape)

                    # Save the output to an array of outputs
                    output_archive.append(output)

                    # Set the input for the layer after
                    next_layer_input = output

                # Find the correct delta value to input into the next layer
                if self.activation_function == "sigmoid":
                    delta = -(expected_output - output) * output * ([1,1,1] - output)
                elif self.activation_function == "relu":
                    for idx, value in enumerate(output):
                        if value > 0:
                            delta[idx] = -(expected_output[idx] - output[idx])
                        else:
                            delta[idx] = 0
                elif self.activation_function == "lrelu":
                    for idx, value in enumerate(output):
                        if value > 0:
                            delta[idx] = -(expected_output[idx] - output[idx])
                        else:
                            delta[idx] = -(expected_output[idx] - output[idx]) * alpha
                else:
                    print("model invalid")
                    break

                # Calculate the squared error for this data point
                squared_error = (expected_output - output)**2
                total_squared_error = total_squared_error + squared_error

                # Find if the prediction was correct for the data point
                prediction = self.get_prediction(output, 0.3) # within 20% of the correct answer

                # Check if prediction is accurate
                if prediction == expected_output:
                    prediction_accurate_for_epoch[str(input_data)].append(True)
                    num_correct_predictions += 1
                else:
                    prediction_accurate_for_epoch[str(input_data)].append(False)

                # Set delta for the second pass
                next_layer_delta = delta

                # Create a list to store the trained the weights and biases
                trained_weights_archive = []
                trained_biases_archive = []

                # Do a second pass through the array (backwards) to update the weights and biases
                for layer_weights, layer_biases, outputs in zip(self.weights[::-1], self.biases[::-1], output_archive[:-1][::-1]):

                    # Calculate the weight and bias change based on the delta we calculated
                    [dummy_output, delta, trained_weights, trained_biases] = self.__array(layer_weights, layer_biases, outputs, next_layer_delta, self.learning_rate, self.activation_function, self.alpha, layer_weights.shape)

                    # Store the trained weights and bias data
                    trained_weights_archive.append(trained_weights)
                    trained_biases_archive.append(trained_biases)

                    next_layer_delta = delta

                # Update the untrained weights to the trained weights for the next iteration
                # The archives are reversed because the second pass was backwards
                self.weights = trained_weights_archive[::-1]
                self.biases = trained_biases_archive[::-1]

            # Calculate the mean squared error for the epoch
            num_data_points = len(input_data_points)
            if num_data_points > 0: # More than 1 data point
                mean_squred_error = total_squared_error/num_data_points

            # Save the mean squared error
            mse_archive.append(mean_squred_error[0])

            # Save the number of correct predictions
            number_of_accurate_predictions_per_epoch.append(num_correct_predictions)

        return self.weights, self.biases, mse_archive, number_of_accurate_predictions_per_epoch

    def __weights_pe(self, delta_k, output_kmin1, partial_sum_out_k, partial_sum_delta_k, init_weight, eta):
        # Compute the calculated output and new weights
        weight = init_weight
        delta = delta_k
        output = output_kmin1
        sum_o = partial_sum_out_k
        sum_s = partial_sum_delta_k
        sum_delta_out = sum_s + delta*weight
        sum_output_out = sum_o + output*weight
        weight_change = weight - output*delta*eta

        return [sum_delta_out, sum_output_out, weight_change]

    def __bias_pe(self, delta_k, sum_in, init_bias, eta):
        # Compute the calculated output and new bias
        net_sum = init_bias + sum_in
        bias_change = init_bias - delta_k*eta
        return [net_sum, bias_change]

    def __activation_pe(self, omega):
        # Calculate the value after the activation function
        if self.activation_function == 'sigmoid':
            return (1 / (1+math.exp(-omega)))
        elif self.activation_function == 'relu':
            return (max(0, omega))
        elif self.activation_function == 'lrelu':
            if omega >= 0:
                return omega
            else:
                return (self.alpha * omega)

    def __error_pe(self, output_kmin1, partial_sum_delta_k, model, alpha):
        # Calculate the error to do backpropigation
        if model == 'sigmoid':
            return output_kmin1 * (1 - output_kmin1) * partial_sum_delta_k
        elif model == 'relu':
            if output_kmin1 > 0:
                return partial_sum_delta_k
            else:
                return 0
        elif model == 'lrelu':
            if output_kmin1 > 0:
                return partial_sum_delta_k
            else:
                return alpha * partial_sum_delta_k

    def __array(self, weights, biases, output_kmin1, delta_k, eta, model, alpha, array_size):
        # Calculate an arbirary size array

        # initialize internal arrays and return values at zero
        output_k = np.zeros(array_size[0])
        delta_kmin1 = np.zeros(array_size[1])
        weight_changes = np.zeros(array_size)
        bias_changes = np.zeros(array_size[0])
        partial_delta_sum = np.zeros(array_size[1])

        # iterate through each of the neurons in the array
        # begin by iterating through each row
        for n in range(array_size[0]):

            # partial sum of the ouput starts at 0
            partial_output_sum = 0

            # iterate through each column of weight PEs in the neuron
            for c in range(array_size[1]):
                [partial_delta_sum[c], partial_output_sum, weight_changes[n, c]] = self.__weights_pe(delta_k[n], output_kmin1[c], partial_output_sum, partial_delta_sum[c], weights[n, c], eta)

            # apply the bias PE
            [net_sum, bias_changes[n]] = self.__bias_pe(delta_k[n], partial_output_sum, biases[n], eta)

            # apply the activation function
            output_k[n] = self.__activation_pe(net_sum)

        output_k = np.array(output_k).reshape(array_size[0],1)

        # update the backpropagation outputs
        for c in range(array_size[1]):
            delta_kmin1[c] = self.__error_pe(output_kmin1[c], partial_delta_sum[c], model, alpha)

        return [output_k, delta_kmin1, weight_changes, bias_changes]

    def check_parameter_dimensions(self):

        # Check that the weights array and biases array have the same dimensions

        # Check that there is the same number of layers
        if len(self.weights) != len(self.biases):
            return False

        # Check that the layers have the same number of nodes
        for idx in range(self.layers):

            # Check that the nodes in the weight array is equal
            # to the nodes in the bias array
            if self.weights[idx].shape[0] != self.biases[idx].shape[0]:
                return False

        # If there are the same number of layers and the layers 
        # have the same number of nodes, return true
        return True

    def get_prediction(self, actual_output, percent_incorrect):
        # Determine which value the actual output of the array is closer to (1, 0) based on the percent accuracy
        # percent incorrect is used for relu and leaky relu, it means that the actual output is less that x% off from the expected output
        # For example, if the actual output is 0.9 and the expected output is 1, 0.05 percent incorrect means that the prediction is undefined, 
        #  while 0.1 percent incorrect would make the prediction a 1

        # Set base answer
        prediction = actual_output[0][0]

        # Based on actiavtion function, determine if closer to 1 or zero
        if self.activation_function == "sigmoid":

            # For Sigmoid of the prediction is based on if the output is bigger or smaller than 0.5
            if prediction >= 0.5:
                prediction = 1
            else:
                prediction = 0

        else:

            # other activation faunction should be within a certain accuracy
            if abs(1-prediction) <= percent_incorrect:
                prediction = 1
            elif abs(0-prediction) <= percent_incorrect:
                prediction = 0

        return [[prediction]]

