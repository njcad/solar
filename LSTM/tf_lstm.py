"""
tf_lstm.py

Description: example Long Short Term Memory (LSTM) learning model for solar power prediction given weather data, from tensorflow.
For Stanford CS221.

Date: 22 November 2023

Authors: Nate Cadicamo, Jessica Yang, Riya Karumanchi, Ira Thawornbut

Approach: use tensorflow library to implement LSTM.

Results: 
    epochs: 100, batch size: 20, optimizer: adam, error: MSE
        TEST LOSS PLANT 1: 0.006391435395926237
        TEST LOSS PLANT 2: 0.010809659957885742
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import get_data as GD

# define data file paths
power1 = "../data/Plant_1_Generation_Data.csv"
weather1 = "../data/Plant_1_Weather_Sensor_Data.csv"
power2 = "../data/Plant_2_Generation_Data.csv"
weather2 = "../data/Plant_2_Weather_Sensor_Data.csv"

# get sorted, normalized, split sequential data from get_data.py
X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = GD.sort_data(weather1, power1)
X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = GD.sort_data(weather2, power2)

# create the tensorflow keras LSTM model
model = Sequential()
features = ['DAILY_YIELD', 'MAX_AMBIENT_TEMP', 'MIN_AMBIENT_TEMP', 'MAX_IRRADIATION', 'MIN_IRRADIATION']
sequence_length = 24
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, len(features))))
model.add(Dense(len(features))) 

# compile the model with adam optimizer and mean squared error (MSE) loss
model.compile(optimizer='adam', loss='mse')



### PLANT 1 MODEL TRAINING AND EVALUATION ###

# train the model on plant 1; TODO experiment with different epochs and batch_size
history_1 = model.fit(X_train_1, y_train_1, epochs=100, batch_size=20, validation_data=(X_val_1, y_val_1))
test_loss_1 = model.evaluate(X_test_1, y_test_1)
print(f"\nTEST LOSS PLANT 1: {test_loss_1}\n")

# plot the training loss, validation loss, and test loss
plt.plot(history_1.history['loss'], label='Training Loss')
plt.plot(history_1.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss_1, color='r', linestyle='--', label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



### PLANT 2 MODEL TRAINING AND EVALUATION ###

# train the model on plant 2; TODO experiment with different epochs and batch_size
history_2 = model.fit(X_train_2, y_train_2, epochs=100, batch_size=20, validation_data=(X_val_2, y_val_2))
test_loss_2 = model.evaluate(X_test_2, y_test_2)
print(f"\nTEST LOSS PLANT 2: {test_loss_2}\n")

# plot the training loss, validation loss, and test loss
plt.plot(history_2.history['loss'], label='Training Loss')
plt.plot(history_2.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss_2, color='r', linestyle='--', label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
