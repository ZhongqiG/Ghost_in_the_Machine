"""
02/07/2024
Quick example to show training of the Rice Model
"""

# Import Packages, Classes, and Functions
import time
import pandas as pd
from scipy.io.arff import loadarff
from IPython.display import display
import matplotlib.pyplot as plt
from rice_model_functions import convert_rice_class_to_number, normalize_dataframe
from gim_simulation_16bit import GIM_simulation_16bit

# Start Timer
start_time = time.time()

# Get the data for the Rice Model
raw_data = loadarff('Rice_Cammeo_Osmancik.arff')
df_raw_data = pd.DataFrame(raw_data[0])

## Data Processing
# Scramble the data rows
df_scrambled = df_raw_data.sample(frac=1, random_state=2)

# Change all data to numbers, specifically the class column
class_array_as_number = convert_rice_class_to_number(df_scrambled)

# Normalize all the data
df_normalized = normalize_dataframe(df_scrambled.drop("Class", axis=1))

## Specify Training and Testing Data
# Get trainning data as numpy array
data_array = df_normalized.to_numpy()
expected_output_array = class_array_as_number

# Choose how many data points for training and testing
# Train on first x data points
train_data_points = data_array[:100]
train_expected_outputs = expected_output_array[:100]

# Test on y data points
test_data_points = data_array[1000:1010]
test_expected_outputs = expected_output_array[1000:1010]

## Train the model
# Create simulation object
rice_model_simulation = GIM_simulation_16bit(integer_bits=7, fraction_bits=9, activation_function="relu")
rice_model_simulation.set_random_initial_condition(layer_array=[7,7,7,1]) # Set random initial weights and biases
rice_model_simulation.set_learning_rate(10**-5) # Set learning rate to very small
rice_model_simulation.set_alpha(0.01) # Set learning rate to very small

# Find how accurate the random initial condition is
# Perform testing
test_actual_outputs = rice_model_simulation.test(test_data_points)

# Get the prediction accuracy of test
percent_prediction_correct_before_training = rice_model_simulation.get_prediction_accuracy(test_actual_outputs, test_expected_outputs, how_close=0.5)

# Print the percent of test points correctly predicted before training
print("When ", len(test_data_points), " data points were tested before training, ", percent_prediction_correct_before_training, "% were predicted correctly.")

# Training Function
_, _, mean_squared_error, avg_weight, largest_weight, num_correct_predictions = rice_model_simulation.train(train_data_points, train_expected_outputs, num_iteration=200)

## Test the performance of the trained model on new data
test_actual_outputs = rice_model_simulation.test(test_data_points)

# Get the prediction accuracy of test
percent_prediction_correct_after_training = rice_model_simulation.get_prediction_accuracy(test_actual_outputs, test_expected_outputs, how_close=0.5)

# Print the percent of test points correctly predicted after training
print("When ", len(test_data_points), " data points were tested after training, ", percent_prediction_correct_after_training, "% were predicted correctly.")

## Display how long it took to run
print("\nProcess finished in %s seconds " % (time.time() - start_time))

## Plot the Mean Squared Error
# Create the Figure
fig, axes = plt.subplots(3, 1)
fig.suptitle('Changes in Mean Squared Error and the Percent of Correct Predictions during Training\n for 7 integer and 9 fraction bits')

axes[0].plot(mean_squared_error, "--")
axes[0].plot(mean_squared_error, "o")
axes[0].set_ylabel("Mean Squared Error")

# Plot the number of correct predictions during training
percent_correct_predictions = [x/len(train_data_points)*100 for x in num_correct_predictions]

axes[1].plot(percent_correct_predictions, "--")
axes[1].plot(percent_correct_predictions, "o")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Percent of Predictions\nCorrect")

# Plot the average weight

axes[2].plot(largest_weight, "--")
axes[2].plot(largest_weight, "o")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Largest Weight")

# Show both plots
plt.show()
