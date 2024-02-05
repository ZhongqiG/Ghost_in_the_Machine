'''
Maya Borowicz, using code developed by Davis Jackson and James Ding
01/31/2024

This Class can be used to simulate the training and testing
of data objects using the GIM architecture.
'''

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
from gim_simulation import GIM_simulation

# Set random seed to keep random numbers consistent (if desired)
np.random.seed(2)

'''
Create a simulation object with weights and biases, this can be done 2 ways
'''
# First, the weights and biases can be specified

# Create the weight and bias arrays, each item in the list represents the array to go from one layer to the next
# Each array should be a np.array object
initial_weights = [np.random.rand(3,2), np.random.rand(1,3)]
initial_biases = [np.random.rand(3,1), np.random.rand(1,1)]

my_simulation_1 = GIM_simulation("relu", initial_weights, initial_biases)

my_simulation_2 = GIM_simulation("relu")
my_simulation_2.set_initial_condition(initial_weights, initial_biases)

# Second, random weights can be generated using the set_random_initial_condition function
# This function inputs a list of the number of nodes in the array

my_simulation_3 = GIM_simulation("relu")
my_simulation_3.set_random_initial_condition([2,3,1], number_times_generated=1)
# This would represent a model with 2 input nodes, one hidden layer with 3 nodes, and an output layer with 2 nodes
# number_times_generated can be used to get the next iteration of weights

# Other paramaters like the activation function, alpha, and the learning rate can also be changed
my_simulation_1.set_activation_function("relu")
my_simulation_1.set_alpha(0.01)
my_simulation_1.set_learning_rate(0.4)

'''
Acquire the input data points. Change it to a np.array object
'''
# Create/acquire input data and expected output data, this data is for the XOR problem
input_data_points = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]
expected_output_points = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]

'''
Observe the inference (testing) of the neural network with the current 
weights and biases
'''
# Test all the simulations (they should get the same result)
observed_output_simulation_1 = my_simulation_1.test(input_data_points)
observed_output_simulation_2 = my_simulation_2.test(input_data_points)
observed_output_simulation_3 = my_simulation_3.test(input_data_points)

# Print the outputs
'''
print("Observed outputs:", 
    "\nSimulation 1: ", observed_output_simulation_1,
    "\nSimulation 2: ", observed_output_simulation_2,
    "\nSimulation 3: ", observed_output_simulation_3)
'''

'''
Training the weights and biases with the data points and their known outputs
'''
# Set the input data, expected output, and number of iterations 
trained_weights, trained_biases, mean_squared_error, num_correct_predictions = my_simulation_1.train(input_data_points, expected_output_points, num_iteration=1000)

'''
Observing the Mean Squared Error per Epoch
'''
# Plot the Mean Squared Error
plt.plot(mean_squared_error, "-")
plt.plot(mean_squared_error, "o")
plt.title("Mean Squared Error per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show() # comment this out to not show the plot

'''
Observing the Number of Correct Predictions
'''
# Plot the Number of correct predictions
plt.plot(num_correct_predictions, "-")
plt.plot(num_correct_predictions, "o")
plt.title("Number of Correct Predictions per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of Correct Predictions")
plt.show()


