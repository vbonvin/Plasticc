import util
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from ANN_bruteforce.DataProcessing import *
from ANN_bruteforce.ResultAnalysis import predict_lightcurves
from util import *
from sklearn.preprocessing import QuantileTransformer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import RNN
from keras.layers import Dropout
import keras.backend as K
from itertools import product
from functools import partial
from tensorflow.keras.models import load_model
from sklearn.externals import joblib

def compute_weights(N_of_Classes, penalty_factor=20):
    "Creating Weights: the critical and arbitrary factor is the penalty_factor"
    w_array = np.ones((len(N_of_Classes)+1, len(N_of_Classes)+1))

    for c_p, c_t in product(range(len(w_array)-1), range(len(w_array)-1)):
        w_array[c_t, c_p] = 1 + N_of_Classes[c_p]/np.sum(N_of_Classes) * penalty_factor        #Penalising 1 False classified as 0 -> has to be higher than 1
    return w_array


"Taken from https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/"

# Recurrent Network
model_name = "LSTM_P0_Feat50_B0E0"

"""  Load Data """
init_data = util.readpickle('../training_samples_astest.pkl')
data_start = init_data.loc[1000:].copy(deep=True)
test = init_data.loc[:1000].copy(deep=True)

#Dataset augmentation
data = dataset_augmentation(data_start, bootstrapping=0, epurate=0)


""" Data Preparation """
## What type of data input needed ?
## (None,  Number of variables, Number of flux sample,) for LSTM
## Number of flux samples  6 errors, 6 fluxes (maybe mjds)
## https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

[zp_data, labels] = dataset_zeropadding_3D(data)
[x_test, y_test] = dataset_zeropadding_3D(test)

scaler = []
for ii in range(zp_data.shape[1]):
    scaler.append(QuantileTransformer(output_distribution='uniform').fit(zp_data[:,ii,:]))
    zp_data[:, ii, :] = scaler[ii].transform(zp_data[:, ii, :])
    x_test[:, ii, :] = scaler[ii].transform(x_test[:, ii, :])

joblib.dump(scaler, model_name + '_scaler.pkl')

x_train = zp_data
y_train = labels

##Convert labels to categorical one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=15)
y_test = keras.utils.to_categorical(y_test, num_classes=15)

""" Loss function definition """
##Personalised loss function , example taken from https://github.com/keras-team/keras/issues/2115
##Test needed comparing categorical cross entropy to weigthed cat cross entropy (without weights)

## Applies weight to categorical cross entropy
def w_categorical_crossentropy(y_true, y_pred, weights):
    "Categorical crossentropy loss function with weights, has to be initalised after with somethin like: "
    "loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)"
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p], K.floatx()) * K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_true, y_pred) * final_mask

## Creating Weights: the critical and arbitrary factor is the penalty_factor
N_of_Classes = np.array([151., 495., 924., 1193., 183., 30., 484., 102., 981.,  208., 370.,  2313.,  239., 175.])
penalty_factor = 0

w_array = compute_weights(N_of_Classes, penalty_factor=penalty_factor)

loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)

""" Model """

model = Sequential()
model.add(LSTM(50, input_shape=(80, zp_data.shape[2])))
model.add(Dropout(0.5))
#model.add(LSTM(20))
#model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))
rmsprop = keras.optimizers.RMSprop(lr=0.0001, decay=0.01)
model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/' + model_name, histogram_freq=0,
                                         write_graph=True, write_images=True)
#Continue Training
#interrupted_model_name = "LSTM_P20_Feat50_B0E0_Number3"
#model.load_weights(interrupted_model_name + '_model_weights.h5')

modelcheckpointCallBack = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',
                                     save_weights_only=True, period=10)

history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), batch_size=32, verbose=1,callbacks = [tbCallBack,modelcheckpointCallBack])

# serialize model to JSON
model_json = model.to_json()
with open(model_name + "_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(model_name + "_model_weights.h5")
#model.load_weights(model_name + '_model_weights.h5')
#del model  # deletes the existing model

#model = keras.models.load_model(model_name, custom_objects={'loss': loss})


""" Visualisation """

score = model.evaluate(x_test, y_test, batch_size=10)
print("Score: ", score)

y_test = predict_lightcurves(model, x_test, y_test)
print(y_test)
plt.show()


"""
Next:
Test Standardisation ('normal')
Test +ANN
Test Lower decay 0.005
Test + z
"""