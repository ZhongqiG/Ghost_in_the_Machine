'''
Maya Borowicz
02/05/2024

This file simulates the training of the Rice Model using floating point numbers.
'''

# Import Packages
import pandas as pd
import numpy as np
import os
import shutil
import time
import matplotlib.pyplot as plt
from gim_simulation import GIM_simulation
from scipy.io.arff import loadarff
from IPython.display import display

# Start Timer
start_time = time.time()

use_saved_weights = False

if not use_saved_weights:
  if os.path.exists('saved_data'):
    shutil.rmtree('saved_data')
  os.mkdir('saved_data')

'''
Functions
'''

def format_rice_data(filename, scrambled=False):
  # Import the data
  raw_data = loadarff(filename)
  df_raw = pd.DataFrame(raw_data[0])

  # Scramble the data
  if scrambled:
    df_raw = df_raw.sample(frac=1, random_state=2)

  # Convert the Class to a 0 or a 1
  class_list = {"Class":[]}
  for value in enumerate(df_raw["Class"]):
    if 'Cammeo' in str(value):
      class_list["Class"].append(0)
    elif 'Osmancik' in str(value):
      class_list["Class"].append(1)

  class_df = pd.DataFrame(class_list)

  # Normalization
  df_dropped= pd.DataFrame.copy(df_raw.drop("Class", axis=1), deep=True)
  df_normalized = (df_dropped - df_dropped.min())/(df_dropped.max() - df_dropped.min())

  # Convert to an array
  data_array = df_normalized.to_numpy()
  expected_output_array = class_df.to_numpy()

  return data_array, expected_output_array

'''
Data Processing
'''

data_array, expected_output_array = format_rice_data('Rice_Cammeo_Osmancik.arff', scrambled=True)


'''
Selecting the Data Points
'''

# Convert to numpy array
num_points = 5

# Select the first x points for training
train_data_points = data_array[:num_points]
train_expected_outputs = expected_output_array[:num_points]

# Select the next x points for testing
test_data_points = data_array[num_points:num_points+100]
test_expected_outputs = expected_output_array[num_points:num_points+100]

# Analysis of training data
num_ones = 0
num_zeros = 0
for value in train_expected_outputs:
  if value == 1:
    num_ones += 1
  elif value == 0:
    num_zeros += 1

print("There are ", num_ones, " ones in the testing data")
print("There are ", num_zeros, " zeros in the testing data")


'''
Creating the Simulation
'''

# Create the simulation object
my_simulation = GIM_simulation("relu")
my_simulation.set_alpha(0.01)
my_simulation.set_learning_rate(10**-5)


# Generate random weights
if not use_saved_weights:
  my_simulation.set_random_initial_condition([7,7,7,1], number_times_generated=1)
  my_simulation.set_alpha(0.01)
  my_simulation.set_learning_rate(10**-5)

# Or used weights saved from a previous training
# Retrieve the saved weights
if use_saved_weights:

  # Create a container for the saved weights and biases
  saved_weights = []
  saved_biases = []

  # Check that saved_data folder exists
  if os.path.exists("saved_data") :
    
    for idx in range(1, 100):
      # Find layer files
      if os.path.exists("saved_data/weights for layer "+str(idx)) and  os.path.exists("saved_data/biases for layer "+str(idx)):
        saved_weights.append(np.loadtxt("saved_data/weights for layer "+str(idx)))
        saved_biases.append(np.loadtxt("saved_data/biases for layer "+str(idx)))
      
      else:
        # There no layer of this index
        break

    # Set the initial condition to use the saved weights and biases
    my_simulation.set_initial_condition(saved_weights, saved_biases)

  else:
    print("ERROR: No data Saved")


'''
Train the data
'''

# train the simulation
# if using old weights, do not train for new ones
if not use_saved_weights:
  trained_weights, trained_biases, mean_squared_error, max_delta, num_correct_predictions = my_simulation.train(train_data_points, train_expected_outputs, num_iteration=300)

# Save the trained weights to a file
if not use_saved_weights:
  for idx in range(len(trained_weights)):
    np.savetxt("saved_data/weights for layer "+str(idx+1), trained_weights[idx])
    np.savetxt("saved_data/biases for layer "+str(idx+1), trained_biases[idx])

# test the accuracy of the trained weights and biases
actual_outputs = my_simulation.test(train_data_points)

# See how accuracte the ouptuts are when run on tested data
correct_predictions = 0
num_zero_predictions = 0
num_zero_actual_outputs = 0
num_zero_expected_outputs = 0

if len(actual_outputs) != 0:
  for idx in range(len(actual_outputs)):
    prediction = my_simulation.get_prediction([[actual_outputs[idx]]], 0.5)

    if prediction == [[0]]:
      num_zero_predictions += 1

    if actual_outputs[idx] == [[0]]:
      num_zero_actual_outputs += 1

    if train_expected_outputs[idx] == [[0]]:
      num_zero_expected_outputs += 1

    # Is the prediction is the same as the expected?
    if prediction == train_expected_outputs[idx]:
      correct_predictions += 1

print("The percent of predictions correct for tested data is ", correct_predictions/len(train_data_points)*100, " %")
print("Number of points that should have been zero: ", num_zero_expected_outputs, 
      "\nNumber of points where the actual output was zero: ", num_zero_actual_outputs, 
      "\nNumber of points where the prediction was zero: ",num_zero_predictions)

## Display how long it took to run
print("\nProcess finished in %s seconds " % (time.time() - start_time))

'''
Plotting
'''

# Plot the Mean Squared Error
if not use_saved_weights:
  fig, axes = plt.subplots(3, 1)

  fig.suptitle('Changes in Mean Squared Error and \nthe Percent of Correct Predictions during Training')

  axes[0].plot(mean_squared_error, "--")
  axes[0].plot(mean_squared_error, "o", markersize=4)
  axes[0].set_ylabel("Mean\n Squared Error")
  axes[0].set_ylim([0,0.5])

  # Plot the number of correct predictions during training
  percent_correct_predictions = [x/len(train_data_points)*100 for x in num_correct_predictions]

  axes[1].plot(percent_correct_predictions, "--")
  axes[1].plot(percent_correct_predictions, "o", markersize=4)
  axes[1].set_xlabel("Epoch")
  axes[1].set_ylabel("Percent of\n Predictions Correct")

  # Plot the average weight value
  axes[2].plot(max_delta, "--")
  axes[2].plot(max_delta, "o", markersize=4)
  axes[2].set_xlabel("Epoch")
  axes[2].set_ylabel("Maximum\n Delta Value")
  print("The maximum delta value was", max(max_delta))

  # Show both plots
  plt.show()
  