from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling1D
from keras.layers import LSTM
# from tensorflow.keras.layers import Conv1D 

predictonModel = Sequential()

predictonModel.add(Conv1D(64, kernel_size=3, activation='relu', padding='causal', input_shape=(20, 1) ))
predictonModel.add(BatchNormalization())
predictonModel.add(Conv1D(64, kernel_size=3, activation='relu', padding='causal'))
predictonModel.add(BatchNormalization())
predictonModel.add(Conv1D(32, kernel_size=3, activation='relu', padding='causal'))
predictonModel.add(MaxPooling1D(pool_size=3)) 
predictonModel.add(LSTM(100 ,  return_sequences=True)) 
predictonModel.add(Dropout(0.2))
predictonModel.add(LSTM(50  , ))
predictonModel.add(Dropout(0.3))
predictonModel.add(Dense(1 , activation='relu'))