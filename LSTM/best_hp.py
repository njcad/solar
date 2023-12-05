"""
best_hp.py

Description: file for optimizing hyperparameters of tensorflow LSTM model with relu activation.

Date: 24 November 2023

Authors: Nate Cadicamo, Jessica Yang, Riya Karumanchi, Ira Thawornbut

Approach:
    1. Get sorted, normalized, clean data from get_data.py
    2. Train tensorflow LSTM model on varying hyperparameters for different number of hidden layers, epoch size, and batch size
    3. Track values of MSE 
    4. Store in CSV files for plant 1 and plant 2
    5. NOTE: this script takes several hours to run. 

Results: (see hp_1.csv, hp_2.csv, hp_analysis.py)
    BEST RESULT 1: {'num_layers': 70, 'num_epochs': 160, 'batch_size': 10, 'test_loss': 0.004714641720056534}
    BEST RESULT 2: {'num_layers': 30, 'num_epochs': 140, 'batch_size': 10, 'test_loss': 0.009615350514650345}

Old results: 
    BEST RESULT FOR PLANT 1: {'num_layers': 40, 'num_epochs': 160, 'batch_size': 10, 'test_loss': 0.004654615186154842}
    BEST RESULT FOR PLANT 2: {'num_layers': 100, 'num_epochs': 160, 'batch_size': 50, 'test_loss': 0.008940218016505241}

"""

import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import get_data as GD

# define data file paths
power1 = "../data/Plant_1_Generation_Data.csv"
weather1 = "../data/Plant_1_Weather_Sensor_Data.csv"
power2 = "../data/Plant_2_Generation_Data.csv"
weather2 = "../data/Plant_2_Weather_Sensor_Data.csv"

# get sorted, normalized, split sequential data from get_data.py
X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = GD.sort_data(weather1, power1)
X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = GD.sort_data(weather2, power2)

# define tf LSTM parameters 
features = ['DAILY_YIELD', 'MAX_AMBIENT_TEMP', 'MIN_AMBIENT_TEMP', 'MAX_IRRADIATION', 'MIN_IRRADIATION']
sequence_length = 24

# keep track of list of results:
results_1 = []
results_2 = []


# define a model-training function to call in loop
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, num_layers, num_epochs, batch_size):
    
    # initialize the model with given values for input
    model = Sequential()
    model.add(LSTM(num_layers, activation='relu', input_shape=(sequence_length, len(features))))
    model.add(Dense(len(features)))
    
    # compile the model with adam optimizer and mean squared error (MSE) loss
    model.compile(optimizer='adam', loss='mse')

    # train the model
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # get and return the test loss
    test_loss = model.evaluate(X_test, y_test)
    return test_loss


# loop over different numbers of hidden layers h, step by 10
for num_layers in range(10, 101, 10):

    # loop over different numbers of epochs:
    for num_epochs in range(20, 201, 10):

        # loop over different batch sizes:
        for batch_size in range(10, 101, 10):

            # train model and get associated test loss for plant 1
            test_loss_1 = train_model(X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1, num_layers, num_epochs, batch_size)

            # train model and get associated test loss for plant 2
            test_loss_2 = train_model(X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2, num_layers, num_epochs, batch_size)
            
            # store the results in a dictionary
            result_1 = {
                'num_layers': num_layers,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'test_loss': test_loss_1
            }
            result_2 = {
                'num_layers': num_layers,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'test_loss': test_loss_2
            }

            # keep track of results
            results_1.append(result_1)
            results_2.append(result_2)


# save the results to csv and also print out the best results
def print_and_save_results(best_result, results, results_file):
    print(f'BEST RESULT: {best_result}\n')
    with open(results_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['num_layers', 'num_epochs', 'batch_size', 'test_loss'])
        writer.writeheader()
        writer.writerows(results)

# print and save results for plant 1
best_result_1 = min(results_1, key=lambda x: x['test_loss'])
print_and_save_results(best_result_1, results_1, "hp_1.csv")

# print and save results for plant 2
best_result_2 = min(results_2, key=lambda x: x['test_loss'])
print_and_save_results(best_result_2, results_2, "hp_2.csv")
