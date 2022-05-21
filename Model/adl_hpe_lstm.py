import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("AAPL_daily_update.csv")
data.head()

#feature extraction
data_final = data[["Date", "Open", "Close"]]
data_final.head()

#visualisation
plt.figure(figsize=(24, 8))
plt.subplot(1, 2, 1)
plt.title('OPEN')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(data_final['Open'], color = 'red')
plt.subplot(1, 2, 2)
plt.title('CLOSE')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(data_final['Close'], color = 'green')
plt.show()

scaler = MinMaxScaler()
print(data_final.loc[1, "Date"].split())
data_final['Date'] = pd.to_datetime(data_final['Date'].apply(lambda x: x.split()[0]))
data_final.set_index('Date', drop = True, inplace = True) 
data_final.loc[:, ["Open", "Close"]] = scaler.fit_transform(data_final.loc[:, ["Open", "Close"]])
data_final.head()

#Train - Test split
train_len = round(len(data_final) * 0.7)
training_data = data_final[:train_len]
training_data.head()

testing_data = data_final[train_len:]
testing_data.head()

#Create training and testing seqeunces along with their prediction labels
def makeSeq(dataset):
  sequences = []
  labels = []
  starting = 0

  for stopping in range(50, len(dataset)): 
    sequences.append(dataset.iloc[starting : stopping])
    labels.append(dataset.iloc[stopping])
    starting += 1
  return (np.array(sequences).astype('float32'), np.array(labels).astype('float32'))

training_seq, training_labels = makeSeq(training_data)
testing_seq, testing_labels = makeSeq(testing_data)

print(np.shape(training_seq), np.shape(training_labels))
print(np.shape(testing_seq), np.shape(testing_labels))
print(testing_seq[0])

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#LSTM Model
model = Sequential()
model.add(LSTM(50, input_shape = (training_seq.shape[1], training_seq.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(2))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = 'mean_absolute_error')
model.summary()

history = model.fit(training_seq, training_labels, epochs = 80, verbose = 1, batch_size = 16)

plt.figure(figsize = (12, 8))
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

labels_predicted = model.predict(testing_seq)
labels_predicted = scaler.inverse_transform(labels_predicted)
plt.figure(figsize = (12, 8))
plt.plot(scaler.inverse_transform(testing_labels), color = 'blue', label = 'Actual')
plt.plot(labels_predicted , color = 'red', label = 'Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

score = model.evaluate(testing_seq, testing_labels, verbose = 1)

print(labels_predicted)

print(scaler.inverse_transform(testing_labels))

import joblib
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
model.save('model.h5')
