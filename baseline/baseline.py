"""
baseline.py

Description: baseline learning model for solar power prediction given weather data. For Stanford CS221.

Date: 16 November 2023

Authors: Nate Cadicamo, Jessica Yang, Riya Karumanchi, Ira Thawornbut

Approach:
    1. Extract features from weather data for each day. Take high_temp, low_temp, and high_irradiation values.
    2. Associated output value will be the maximum total daily power yield from that day
    3. Use scikit-learn linear regression

Results: 
    Learned weights for plant1: [ 211.68077902  -74.64520767 3210.23206367]
    Bias term for plant1: -729.282971074088
    R-squared error for training set on plant1: 0.2902183018575516
    R-squared error for test set on plant1: 0.3118266568343654

    Learned weights for plant2: [ 408.17239839  -19.15427342 1241.55667972]
    Bias term for plant2: -6638.3181962395265
    R-squared error for training set on plant2: 0.7631709368546855
    R-squared error for test set on plant2: 0.7093351479346417
"""

import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# get csv data files (may add more later)
power1 = "../data/Plant_1_Generation_Data.csv"
weather1 = "../data/Plant_1_Weather_Sensor_Data.csv"
power2 = "../data/Plant_2_Generation_Data.csv"
weather2 = "../data/Plant_2_Weather_Sensor_Data.csv"

# put into lists to iterate over in case we want to add more later
weather_data = [weather1, weather2]
power_data = [power1, power2]

# iterate over the two data files
for i in range(1, len(weather_data) + 1):

    # get local weather and generation data
    this_weather, this_power = weather_data[i - 1], power_data[i - 1]

    # read weather data into pandas dataframe
    df_weather = pd.read_csv(this_weather)

    # convert date_time column name in df to actual datetime
    df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])
    df_weather['DATE'] = df_weather['DATE_TIME'].dt.date

    # sort data by date (not time) : min and max temp, max irradiation
    df_weather = df_weather.groupby('DATE').agg({
        'AMBIENT_TEMPERATURE': ['max', 'min'],
        'IRRADIATION': 'max'
    }).reset_index()

    # reset the columns
    df_weather.columns = ['DATE', 'MAX_AMBIENT_TEMP', 'MIN_AMBIENT_TEMP', 'MAX_IRRADIATION']

    # read power data into pandas dataframe
    df_power = pd.read_csv(this_power)
     
    # convert date_time column name in df to actual datetime
    df_power['DATE_TIME'] = pd.to_datetime(df_power['DATE_TIME'])
    df_power['DATE'] = df_power['DATE_TIME'].dt.date

    # sort data by date (not time) : max daily power yield
    df_power = df_power.groupby('DATE').agg({
        'DAILY_YIELD': 'max'
    })

    # merge weather and power dataframs
    merged_df = pd.merge(df_power, df_weather, on='DATE')

    # extract X matrix and y vector 
    X = merged_df[['MAX_AMBIENT_TEMP', 'MIN_AMBIENT_TEMP', 'MAX_IRRADIATION']]
    y = merged_df['DAILY_YIELD']

    # split the dataframes into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # use a linear regression model from scikit-learn
    model = LinearRegression()
    model.fit(X_train, y_train)

    # get the actual learned weight vector and bias term
    print(f'Learned weights for plant{i}: {model.coef_}')
    print(f'Bias term for plant{i}: {model.intercept_}')

    # get y predictions for training set and test set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # get percent differences between value and prediction for graphs
    abs_differences_train = abs((y_pred_train - y_train) / y_train) * 100
    abs_differences_test = abs((y_pred_test - y_test) / y_test) * 100
    print(f'Mean absolute percent difference for training set on plant{i}: {abs_differences_train.mean()}')
    print(f'Mean absolute percent difference for test set on plant{i}: {abs_differences_test.mean()}')

    # get r_squared error
    r_squared_train = r2_score(y_train, y_pred_train)
    r_squared_test = r2_score(y_test, y_pred_test)
    print(f'R-squared error for training set on plant{i}: {r_squared_train}')
    print(f'R-squared error for test set on plant{i}: {r_squared_test}')

    # plot the differences for training set
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(abs_differences_train)), abs_differences_train)
    plt.title(f'Percentage Difference Between Predicted and Actual Values: training set, plant{i}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Percentage Difference')
    plt.ylim(0, 100)
    plt.savefig(f'train_plant_{i}.png')

    # plot the differences for test set
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(abs_differences_test)), abs_differences_test)
    plt.title(f'Percentage Difference Between Predicted and Actual Values: test set, plant{i}')
    plt.xlabel('Data Point Index')
    plt.ylabel('Percentage Difference')
    plt.ylim(0, 100)
    plt.savefig(f'test_plant_{i}.png')
