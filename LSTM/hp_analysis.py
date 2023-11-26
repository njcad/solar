"""
hp_analysis.py

Description: file for analyzing hyperparameter optimization of tensorflow LSTM model with relu activation (see best_hp.py).

Date: 25 November 2023

Authors: Nate Cadicamo, Jessica Yang, Riya Karumanchi, Ira Thawornbut

Approach:
    1. In best_hp.py, we saved data about test loss into hp_1.csv and hp_2.csv
    2. Reference these csv files to find top 25 parameter combinations for each plant
    3. Plot these values, along with their averages. Save to plant_1_hp.png and plant_2_hp.png
    4. Record results (see below)

Results: (see hp_1.csv, hp_2.csv for raw data, plant_1_hp.png, plant_2_hp.png for graphs)

    Overall best results given by best_hp.py
    BEST RESULT 1: {'num_layers': 70, 'num_epochs': 160, 'batch_size': 10, 'test_loss': 0.004714641720056534}
    BEST RESULT 2: {'num_layers': 30, 'num_epochs': 140, 'batch_size': 10, 'test_loss': 0.009615350514650345}

    Average values on best 25 values for plant 1:
    num_layers     65.200000
    num_epochs    155.600000
    batch_size     23.200000
    test_loss       0.004897

    Average values on best 25 values for plant 2:
    num_layers     64.800000
    num_epochs    144.400000
    batch_size     16.400000
    test_loss       0.010472

"""

import pandas as pd
import matplotlib.pyplot as plt

# load the csv data files into pandas dataframes
df_1 = pd.read_csv('hp_1.csv')
df_2 = pd.read_csv('hp_2.csv')

# sort the dataframes by the 'test_loss' column in ascending order
df_1_sorted = df_1.sort_values(by='test_loss')
df_2_sorted = df_2.sort_values(by='test_loss')


### PLANT 1 DATA ANALYSIS ###


# get the top 25 rows by minimum test loss for dataset 1
top_rows_1 = df_1_sorted.head(25)

# find the average values for each hyperparameter
average_values_1 = top_rows_1.mean()
print(f'Average values for plant 1:')
print(average_values_1)

# plot the parameters for dataset 1
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# plot for num_layers
axes[0, 0].bar(range(len(top_rows_1)), top_rows_1['num_layers'], align='center')
axes[0, 0].axhline(y=average_values_1['num_layers'], color='r', linestyle='--', label='Average')
axes[0, 0].set_title('num_layers (Plant 1)')

# plot for num_epochs
axes[0, 1].bar(range(len(top_rows_1)), top_rows_1['num_epochs'], align='center')
axes[0, 1].axhline(y=average_values_1['num_epochs'], color='r', linestyle='--', label='Average')
axes[0, 1].set_title('num_epochs (Plant 1)')

# plot for batch_size
axes[1, 0].bar(range(len(top_rows_1)), top_rows_1['batch_size'], align='center')
axes[1, 0].axhline(y=average_values_1['batch_size'], color='r', linestyle='--', label='Average')
axes[1, 0].set_title('batch_size (Plant 1)')

# plot for test_loss
axes[1, 1].bar(range(len(top_rows_1)), top_rows_1['test_loss'], align='center')
axes[1, 1].axhline(y=average_values_1['test_loss'], color='r', linestyle='--', label='Average')
axes[1, 1].set_title('test_loss (Plant 1)')

# plot and save
plt.tight_layout()
plt.savefig('plant_1_hp.png')
plt.show()


### PLANT 2 DATA ANALYSIS ###


# get the top 25 rows by minimum test loss for dataset 2
top_rows_2 = df_2_sorted.head(25)

# find the average values for each hyperparameter
average_values_2 = top_rows_2.mean()
print(f'Average values for plant 2:')
print(average_values_2)

# plot the parameters for dataset 1
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# plot for num_layers
axes[0, 0].bar(range(len(top_rows_2)), top_rows_2['num_layers'], align='center')
axes[0, 0].axhline(y=average_values_2['num_layers'], color='r', linestyle='--', label='Average')
axes[0, 0].set_title('num_layers (Plant 2)')

# plot for num_epochs
axes[0, 1].bar(range(len(top_rows_2)), top_rows_2['num_epochs'], align='center')
axes[0, 1].axhline(y=average_values_2['num_epochs'], color='r', linestyle='--', label='Average')
axes[0, 1].set_title('num_epochs (Plant 2)')

# plot for batch_size
axes[1, 0].bar(range(len(top_rows_2)), top_rows_2['batch_size'], align='center')
axes[1, 0].axhline(y=average_values_2['batch_size'], color='r', linestyle='--', label='Average')
axes[1, 0].set_title('batch_size (Plant 2)')

# plot for test_loss
axes[1, 1].bar(range(len(top_rows_2)), top_rows_2['test_loss'], align='center')
axes[1, 1].axhline(y=average_values_2['test_loss'], color='r', linestyle='--', label='Average')
axes[1, 1].set_title('test_loss (Plant 2)')

# plot and save
plt.tight_layout()
plt.savefig('plant_2_hp.png')
plt.show()
