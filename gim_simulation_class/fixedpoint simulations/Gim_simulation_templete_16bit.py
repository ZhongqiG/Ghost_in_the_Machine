'''
Maya Borowicz, using code developed by Davis Jackson and James Ding
02/05/2024

This Class can be used to simulate the training and testing
of data objects using the GIM architecture with fixed point values.
'''

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from gim_simulation_16bit import GIM_simulation_16bit
from fixedpoint.fixedpoint import FixedPoint


# Set random seed to keep random numbers consistent (if desired)
np.random.seed(2)

'''
Create a simulation object with weights and biases, this can be done 2 ways
'''
# First, the weights and biases can be specified

# Create the weight and bias arrays, each item in the list represents the array to go from one layer to the next
# Each array should be a np.array object
times_generated = 2
for i in range(times_generated):
    initial_weights = [list(np.random.rand(3,2)), list(np.random.rand(1,3))]
    initial_biases = [list(np.random.rand(3,1)), list(np.random.rand(1,1))]

my_simulation_1 = GIM_simulation_16bit(integer_bits=5, fraction_bits=11, activation_function="relu", weight_array=initial_weights, bias_array=initial_biases)

my_simulation_2 = GIM_simulation_16bit(integer_bits=5, fraction_bits=11, activation_function="relu")
my_simulation_2.set_initial_condition(initial_weights,initial_biases)

my_simulation_3 = GIM_simulation_16bit(integer_bits=5, fraction_bits=11, activation_function="relu")
my_simulation_3.set_random_initial_condition(layer_array=[2,3,1], number_times_generated=2)

my_simulation_3.set_alpha(0.1)
my_simulation_3.set_learning_rate(0.1)

'''
Testing of the simulations to ensure the same outputs
'''

# Get the data points for the 2 input XOR
input_data_points = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]
expected_output_points = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]

observed_output_simulation_1 = my_simulation_1.test(input_data_points)
observed_output_simulation_2 = my_simulation_2.test(input_data_points)
observed_output_simulation_3 = my_simulation_3.test(input_data_points)

# Check that both simulations [that have the same weights and biases] 
# have the same outputs when they are tested with the same input data points
if observed_output_simulation_1 == observed_output_simulation_2 and observed_output_simulation_1 == observed_output_simulation_3:
    print("Outputs for simulation 1, 2, and 3 are the same")

else:
    print("The outputs for simulation 1, 2, and 3 are NOT the same")


'''
Now let's move on to training the data
'''
trained_weights, trained_biases, mean_squared_error, num_correct_predictions = my_simulation_3.train(input_data_points, expected_output_points, num_iteration=200)

#print(my_simulation_3.weights)
#print(my_simulation_3.biases)

# Error the weights and biases are all zero????

'''
Get the accuracy of the trained network
'''

# Plot the Mean Squared Error
fig, axes = plt.subplots(2, 1)

fig.suptitle('Changes in Mean Squared Error and the Percent of Correct Predictions during Training\nfor the 2-input XOR with 5 integer and 11 fraction bits')

axes[0].plot(mean_squared_error, "--")
axes[0].plot(mean_squared_error, "o", markersize=2.5)
axes[0].set_ylabel("Mean Squared Error")

# Plot the number of correct predictions during training
percent_correct_predictions = [x/len(input_data_points)*100 for x in num_correct_predictions]

axes[1].plot(percent_correct_predictions, "--")
axes[1].plot(percent_correct_predictions, "o", markersize=2.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Percent of Predictions Correct")

# Show both plots
plt.show()
