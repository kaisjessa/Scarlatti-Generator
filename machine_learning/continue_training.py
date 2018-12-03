#imports
import numpy as np
import pickle
import keras.models
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
#from data_processing import *

def train_model(num_epochs):
	with open('data/models/processed_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
	    _, X_2, _, y_2, _, _, _ = pickle.load(f)

	#Keras NN
	model = keras.models.load_model("data/models/current_model.h5")

	filepath = "data/models/current_model.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	#train the model
	model.fit(X_2, y_2, epochs=num_epochs, batch_size=64, callbacks=callbacks_list)
	#save the model
	model.save("data/models/final_model.h5")
	print("Training Completed!")

train_model(128-29-22-49)
