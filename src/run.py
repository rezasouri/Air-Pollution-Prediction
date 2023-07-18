import numpy as np
import pandas as pd
from sklearn import preprocessing

# this is a function for transform Aotizhongxin dataframe to best form based on the problem in homework
def transformDatasetAotiz(data):
    data = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    dataframe = pd.DataFrame(data_scaled,columns=['year','month','day','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM'])
    return dataframe

# this is a function for transform non Aotizhongxin dataframe to best form based on the problem in homework
def transformDatasetNonAotiz(data):
    name = data.columns[-1] # return name of station
    data = data['PM2.5'].values.reshape(-1,1) #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    dataframe= pd.DataFrame(data_scaled ,columns=[f'PM2.5_{name}'])
    return dataframe

# in this section Aotiz df imported and necessary preprocessing adopted
df_Aotizhongxin = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv')
df_Aotizhongxin=df_Aotizhongxin.interpolate(limit_direction='both')
df_Aotizhongxin['wd']= df_Aotizhongxin['wd'].replace(['N', 'NNE', 'NE', 'ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'],
                                                     [0, 22.5, 45, 67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5])
df_Aotizhongxin=df_Aotizhongxin.drop(['station', 'No'], axis=1)
# print(df_Aotizhongxin)


x = df_Aotizhongxin.values #returns a numpy array
min_max_scaler = pr