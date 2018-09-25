#imports
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
#from data_processing import *

with open('data/models/processed_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    _, X_2, _, y_2, _, _, _ = pickle.load(f)

#Keras NN
model = Sequential()
#add long short term memory cell
#output is 700 units (length), input_shape is (number_of_inputs, input_length), return full sequence
model.add(LSTM(700, input_shape=(X_2.shape[1], X_2.shape[2]), return_sequences=True)) #layer 1
#account for overfitting
model.add(Dropout(0.2))

#add another LSTM layer
model.add(LSTM(700, return_sequences=True)) #layer 2
#account for overfitting
model.add(Dropout(0.2))

#add another LSTM layer
model.add(LSTM(700)) #layer 3
#account for overfitting
model.add(Dropout(0.2))

#Fully connected (dense) output layer
model.add(Dense(y_2.shape[1], activation='softmax')) #layer 4
#minimize loss
model.compile(loss='categorical_crossentropy', optimizer='adam')
print("Model compiled")

filepath = "data/models/current_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#train the model
model.fit(X_2, y_2, epochs=128, batch_size=64, callbacks=callbacks_list)
#save the model
model.save("data/models/final_model.h5")
print("Training Completed!")
