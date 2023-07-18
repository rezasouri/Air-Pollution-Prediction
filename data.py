import numpy as np
import pandas as pd
from sklearn import preprocessing

# this is a function for transform Aotizhongxin dataframe to best form based on the problem in homework
def transformDatasetAotiz(data):
    data = data.interpolate(limit_direction='both')
    data = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    dataframe = pd.DataFrame(data_scaled,columns=['year','month','day','hour','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM'])
    return dataframe

# this is a function for transform non Aotizhongxin dataframe to best form based on the problem in homework
def transformDatasetNonAotiz(data):
    name = data.columns[-1] # return name of station
    data['PM2.5'] = data['PM2.5'].interpolate( limit_direction='both')
    data = data['PM2.5'].values.reshape(-1,1) #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    dataframe = pd.DataFrame(data_scaled ,columns=[f'PM2.5_{name}'])
    return dataframe

# in this section Aotiz df imported and necessary preprocessing adopted
df_Aotizhongxin = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv')
df_Aotizhongxin=df_Aotizhongxin.interpolate(limit_direction='both')
df_Aotizhongxin['wd']= df_Aotizhongxin['wd'].replace(['N', 'NNE', 'NE', 'ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'],
                                                     [0, 22.5, 45, 67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5])
df_Aotizhongxin=df_Aotizhongxin.drop(['station', 'No'], axis=1)
PM25_Aotizhongxin=df_Aotizhongxin['PM2.5']
Aotiz=df_Aotizhongxin.drop(['year','month','day','hour','PM2.5','SO2','NO2','O3'],axis=1)

df_Changping = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Changping_20130301-20170228.csv')
df_Dingling = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv')
df_Dongsi = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv')
df_Guanyuan = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Guanyuan_20130301-20170228.csv')
df_Gucheng = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Gucheng_20130301-20170228.csv')
df_Huairou = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Huairou_20130301-20170228.csv')
df_NongZhanguan = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Nongzhanguan_20130301-20170228.csv')
df_Shunyi = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Shunyi_20130301-20170228.csv')
df_Tiantan = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Tiantan_20130301-20170228.csv')
df_Wanliu = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Wanliu_20130301-20170228.csv')
df_Wanshouxigong = pd.read_csv('/home/rezaz/deeplearningProject/Air-Pollution-Prediction/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Wanshouxigong_20130301-20170228.csv')

PM25_Changping = transformDatasetNonAotiz(df_Changping)
PM25_Dingling = transformDatasetNonAotiz(df_Dingling)
PM25_Dongsi = transformDatasetNonAotiz(df_Dongsi)
PM25_Guanyuan = transformDatasetNonAotiz(df_Guanyuan)
PM25_Gucheng = transformDatasetNonAotiz(df_Gucheng)
PM25_Huairou = transformDatasetNonAotiz(df_Huairou)
PM25_NongZhanguan = transformDatasetNonAotiz(df_NongZhanguan)
PM25_Shunyi = transformDatasetNonAotiz(df_Shunyi)
PM25_Tiantan = transformDatasetNonAotiz(df_Tiantan)
PM25_Wanliu = transformDatasetNonAotiz(df_Wanliu)
PM25_Wanshouxigong = transformDatasetNonAotiz(df_Wanshouxigong)

PM25_result = pd.concat([PM25_Aotizhongxin, PM25_Changping, PM25_Dingling, PM25_Dongsi, PM25_Guanyuan, PM25_Gucheng, PM25_Huairou,
                    PM25_NongZhanguan, PM25_Shunyi, PM25_Tiantan, PM25_Wanliu, PM25_Wanshouxigong], axis=1)

result = pd.concat([PM25_result, Aotiz], axis=1)