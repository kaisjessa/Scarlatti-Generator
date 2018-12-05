#imports
import numpy as np
import pickle
from time import time
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
#from data_processing import *

with open('data/models/d_minor_processed_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    _, X_2, _, y_2, _, _, _ = pickle.load(f)

#Keras NN
model = Sequential()
#add long short term memory cell
#output is 256 units (length), input_shape is (number_of_inputs, input_length), return full sequence
model.add(LSTM(256, input_shape=(X_2.shape[1], X_2.shape[2]), return_sequences=True)) #layer 1
#account for overfitting
model.add(Dropout(0.2))
#add another LSTM cell (layer 2)
model.add(LSTM(256, return_sequences=False))
#account for overfitting
model.add(Dropout(0.2))

#Fully connected (dense) output layer
model.add(Dense(y_2.shape[1], activation='softmax')) #layer 4

#minimize loss
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Model compiled")

#save models after every epoch if the loss improves
filepath = "data/models/new_current_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir="logs/{}".format("D_MINOR_ONLY".format(time())))
#tensorboard --logdir=logs/
callbacks_list = [checkpoint, tensorboard]

#train the model
model.fit(X_2, y_2, epochs=128, batch_size=64, callbacks=callbacks_list)
#save the model
model.save("data/models/d_minor_model.h5")
print("Training Completed!")