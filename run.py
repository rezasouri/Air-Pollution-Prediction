import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from keras import optimizers
import tensorflow as tf
from tensorflow_addons.metrics import RSquare
from data import result, PM25_result
from models import predictonModel

# in problem we should plot pearson correlation of PM data  of all station
# we use seaborn library for this issue
pearsoncorr = PM25_result.corr(method='pearson')
sb.set(rc = {'figure.figsize':(15,8)}) 
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

plt.show()
result = result.fillna(result['wd']. value_counts(). index[0])

# we split data to train and test 
# we should not shuffle train and test data because this is time series data
X_train_1lag = result[0:28052]
X_test_1lag = result[28052:35063]
y_train_1lag = result[1:28053]
y_test_1lag = result[28053:35064]

print(X_train_1lag.shape)
print(y_train_1lag.shape)
print(X_test_1lag.shape)
print(y_test_1lag.shape)

# compiling the model 
predictonModel.compile(loss='mse', 
              optimizer=optimizers.Adam(learning_rate=1e-3, decay=1e-4),
              metrics=[tf.keras.metrics.RootMeanSquaredError(),
                       tf.keras.metrics.MeanAbsoluteError(),
                       RSquare()])
predictonModel.summary()
bach_size=32
epoch=200
# fitting model for 1 lag data
predictonModel.fit(X_train_1lag, y_train_1lag, epochs=200,
          batch_size=bach_size,
          steps_per_epoch=X_train_1lag.shape[0] // bach_size,
          verbose=2,
          validation_split=0.1)

# in this part we prudece 7 lag dataset from old dataset
lags = 7    
concat = []
for i in range(lags):
    concat.append( result.shift(i) )

result_7lag = pd.concat( concat, axis=1)

X_train_7lag=result_7lag[6:28052]
X_test_7lag=result_7lag[28052:35063]
y_train_7lag=result_7lag[7:28053]
y_test_7lag=result_7lag[28053:35064]

print(X_train_7lag.shape)
print(y_train_7lag.shape)
print(X_test_7lag.shape)
print(y_test_7lag.shape)

# fitting model for 1 lag data
predictonModel.fit(X_train_7lag, y_train_7lag, epochs=200,
          batch_size=bach_size,
          validation_split=0.1)