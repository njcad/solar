import pandas as pd 

generation1 = "data/Plant_1_Generation_Data.csv"
weather1 = "data/Plant_1_Weather_Sensor_Data.csv"
generation2 = "data/Plant_2_Generation_Data.csv"
weather2 = "data/Plant_2_Weather_Sensor_Data.csv"

weather_data = [weather1, weather2]
generation_data = [generation1, generation2]
data_features = {}

for data_file in weather_data:

    df = pd.read_csv(data_file)

    features = {}
    for row_tuple in df.iterrows():
        row_idx = row_tuple[0]
        row = row_tuple[1]

        date_time = row['DATE_TIME']
        temp = row['AMBIENT_TEMPERATURE']
        irradiation = row['IRRADIATION']
        
        features[date_time] = (temp, irradiation)

    data_features[data_file] = features

