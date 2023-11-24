"""
get_data.py

Description: sorting solar power and weather data into usable datasets for LSTM model. For Stanford CS221.

Date: 22 November 2023

Authors: Nate Cadicamo, Jessica Yang, Riya Karumanchi, Ira Thawornbut

Approach: 
    1. Sort data with pandas to build feature vectors of max_temp, min_temp, max_irradiation, min_irradiation. 
    2. Split data into training, validation, and testing sets with sizes 0.8, 0.1, 0.1 of original data, in 
    chronological order (since it is time-series).
    3. Normalize data with MinMaxScaler, which scales the data to within [0, 1].
    4. Build 24 hour sequences from sorted, split, and normalized data to feed into LSTM. Calling sort_data() 
    externally will return these sequences.  
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# obtain data from csv files
def sort_data(weather, power):

    # read weather data into pandas df
    df_weather = pd.read_csv(weather)

    # convert date_time column name in df to actual datetime
    # # NOTE: Jessica added "format..." onwards to test lstm.py
    # df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'], format='%d-%m-%Y %H:%M', dayfirst=True)
    df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'])


    # extract date and hour from the datetime
    df_weather['DATE'] = df_weather['DATE_TIME'].dt.date
    df_weather['HOUR'] = df_weather['DATE_TIME'].dt.hour

    # sort data by date and hour: max and min temp, max and min irradiation
    df_weather = df_weather.groupby(['DATE', 'HOUR']).agg({
        'AMBIENT_TEMPERATURE': ['max', 'min'],
        'IRRADIATION': ['max', 'min']
    }).reset_index()

    # reset the columns
    df_weather.columns = ['DATE', 'HOUR', 'MAX_AMBIENT_TEMP', 'MIN_AMBIENT_TEMP', 'MAX_IRRADIATION', 'MIN_IRRADIATION']

    # read power data into pandas dataframe
    df_power = pd.read_csv(power)

    # convert date_time column name in df to actual datetime
    df_power['DATE_TIME'] = pd.to_datetime(df_power['DATE_TIME'])

    # extract date and hour from the datetime
    df_power['DATE'] = df_power['DATE_TIME'].dt.date
    df_power['HOUR'] = df_power['DATE_TIME'].dt.hour

    # sort data by date and hour: max daily power yield
    df_power = df_power.groupby(['DATE', 'HOUR']).agg({
        'DAILY_YIELD': 'max'
    }).reset_index()

    # merge weather and power dataframes
    merged_df = pd.merge(df_power, df_weather, on=['DATE', 'HOUR'])

    # pass over this merged_df to clean_data to get finalized data
    return clean_data(merged_df)


# split and normalize data
def clean_data(df):

    # split data into training, validation, and testing sets
    train_size, other_size = int(len(df) * 0.8), int(len(df) * 0.1)
    train_data = df[:train_size]
    val_data = df[train_size : train_size + other_size]
    test_data = df[train_size + other_size:]

    # normalize the data with MinMaxScaler 
    features = ['DAILY_YIELD', 'MAX_AMBIENT_TEMP', 'MIN_AMBIENT_TEMP', 'MAX_IRRADIATION', 'MIN_IRRADIATION']
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[features])
    val_scaled = scaler.fit_transform(val_data[features])
    test_scaled = scaler.transform(test_data[features])

    # sort data into 24 hour sequences
    sequence_length = 24
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_val, y_val = create_sequences(val_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)

    # return these sequences for deployment by LSTM model
    return X_train, y_train, X_val, y_val, X_test, y_test


# prepare sequences for LSTM model
def create_sequences(data, sequence_length):

    # initialize lists for sequences and labels
    sequences, labels = [], []

    # build sequences of 24 hours 
    for i in range(len(data) - sequence_length):
        seq = data[i : i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)

    # return the X matrix and y vector
    return np.array(sequences), np.array(labels)
