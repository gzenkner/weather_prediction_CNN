# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import seaborn as sns

#1 load data from CSV
df = pd.read_csv('drive/MyDrive/Weather Forecasting/Datasets/kew_heathrow_6y.csv', index_col=0)
df = df.drop(labels='ob_time.1', axis=1)
df.index = pd.to_datetime(df.index, utc=True)
df = df.fillna(df.mean())
train = int(0.7*len(df))
val = int(0.9*len(df))

#2 scale 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.iloc[:train, :])
df_scaled = pd.DataFrame(scaler.transform(df))
df_scaled.columns = df.columns
df_scaled.index = df.index

#3 create window
WINDOW_SIZE = 2

def df_to_X_y(df, window_size=WINDOW_SIZE):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = [r for r in df_as_np[i+window_size]]
    y.append(label)
  return np.array(X), np.array(y)

#4 train, val, test
X_train, y_train = X[:train], y[:train]
X_val, y_val = X[train:val], y[train:val]
X_test, y_test = X[val:], y[val:]

#5 model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

features = len(df.columns)
model = Sequential()
model.add(InputLayer((WINDOW_SIZE, features)))
model.add(LSTM(64))
model.add(Dense(64, 'relu'))
model.add(Dense(32, 'relu'))
model.add(Dropout(0.1))
model.add(Dense(features, 'linear'))
model.summary()

model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

patience = 2
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, callbacks=[early_stopping], batch_size=2)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#6 predict, autoregressive
test_predictions = [] 
out_steps = 24*7
current_batch = X_test[0:1,:,:] #takes first sample, all windows and all features
for i in range(out_steps):
  current_pred = model.predict(current_batch).flatten()
  test_predictions.append(current_pred)
  current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
test_predictions = np.array(test_predictions)

#7 invert scale
y_pred_multi = test_predictions[:, 0] * Temp_2m_training_std3 + Temp_2m_training_mean3
y_test_multi = scaler.inverse_transform(X_train)


#8 plot 
fig, ax = plt.subplots()
ax.scatter(x=indices_out, y=y_pred_multi, label='Test Predictions', c='orange', alpha=0.8, edgecolors='none', s=80, marker='x')
ax.scatter(x=indices_out, y=y_test_multi, label='T (degC)',c='green', alpha=0.5, edgecolors='none', s=80)
ax.scatter(x=indices_in, y=X3_val_eval, label='Input', alpha=0.5, edgecolors='none', s=80)
RMSE = sqrt(mean_squared_error(y_pred_multi, y_test_multi))
MAE = mean_absolute_error(y_pred_multi, y_test_multi)
plt.title('RMSE ' + str(RMSE)+ ': MAE ' + str(MAE))
plt.xlabel('Date Time')
plt.ylabel('Temperature (C)')
ax.legend()
plt.show()

