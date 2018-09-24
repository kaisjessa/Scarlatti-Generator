#imports
import numpy as np
import pandas as pd
import random
import keras.models
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import pickle, sys
from midi_to_txt import text_to_midi
from music21 import converter, instrument, note, chord, stream
#from data_preprocessing import *

with open('data/processed_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    X, _, y, _, chat_to_int, int_to_char, chars = pickle.load(f)

with open('data/training_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    text = pickle.load(f)

text = text[0]
chars = sorted(list(set(text)))

#load model
model = keras.models.load_model("data/test_model.h5")

#sample_char = "" if len(sys.argv) < 2 else sys.argv[1]
#take an array of inputs

def check_model():
    output_array = []
    #random starting point from training data for generation
    int_train = X[random.randint(0, len(X))]
    #convert starting point back to characters
    chars_array = [int_to_char[n] for n in int_train]

    #number of characters to generate
    for i in range(500):
        #reshape data to feed to NN
        x = np.reshape(int_train, (1, len(int_train), 1))
        #normalize for NN
        x = x / float(len(chars))

        #the prediction is the index of the next character index
        #argmax takes the highest number in the onehot array
        int_prediction = np.argmax(model.predict(x, verbose=0))

        #append prediction to string array for output
        chars_array.append(int_to_char[int_prediction])
        output_array.append(int_to_char[int_prediction])

        #append index to index array
        int_train.append(int_prediction)
        #drop first element for next iteration
        int_train = int_train[1:]

    return(output_array)

music_text = check_model()
music_midi = text_to_midi(music_text)
midi_stream = stream.Stream(music_midi)
midi_stream.write('midi', fp='model_output.mid')
print("MIDI saved")
