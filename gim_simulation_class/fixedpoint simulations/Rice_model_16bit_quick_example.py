"""
02/07/2024
Quick example to show training of the Rice Model
"""

# Import Packages, Classes, and Functions
import time
import pandas as pd
from scipy.io.arff import loadarff
from IPython.display import display
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
# Train on first 1000 data points
train_data_points = data_array[:1000]
train_expected_outputs = expected_output_array[:1000]

# Test on 500 data points
test_data_points = data_array[1000:1500]
test_expected_outputs = expected_output_array[1000:1500]

## Train the model
# Create simulation object
rice_model_simulation = GIM_simulation_16bit(integer_bits=5, fraction_bits=11, activation_function="relu")
rice_model_simulation.set_random_initial_condition(layer_array=[7,7,1]) # Set random initial weights and biases
rice_model_simulation.set_learning_rate(10**-3) # Set learning rate to very small

# Training Function
_, _, mean_squared_error, num_correct_predictions = rice_model_simulation.train(train_data_points, train_expected_outputs, num_iteration=10)

## Test the performance of the trained model on new data
test_actual_outputs = rice_model_simulation.test(test_data_points)

# Get the prediction accuracy of test
percent_prediction_correct = rice_model_simulation.get_prediction_accuracy(test_actual_outputs, expected_outputs, how_close=0.2)

## Display how long it took to run
print("Process finished --- %s seconds ---" % (time.time() - start_time))