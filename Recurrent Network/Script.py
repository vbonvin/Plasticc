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
from keras.layers import Dropout

"Taken from https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/"

# Recurrent Network
model_name = "run_RecNN_Feat50_Dropout05_StandY_boot5_epu5_mjdsY_rmsprop_lr0.001_decay0.1"

"""  Load Data """
init_data = util.readpickle('../training_samples.pkl')
data_start = init_data.loc[1000:].copy(deep=True)
test = init_data.loc[:1000].copy(deep=True)

#Dataset augmentation
data = dataset_augmentation(data_start, bootstrapping=5, epurate=5)


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

print("after standardisation",zp_data,zp_data.shape)

x_train = zp_data
y_train = labels

##Convert labels to categorical one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=15)
y_test = keras.utils.to_categorical(y_test, num_classes=15)


""" Model """

model = Sequential()
model.add(LSTM(50, input_shape=(80, zp_data.shape[2])))
model.add(Dropout(0.5))
#model.add(LSTM(20))
#model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))
rmsprop = keras.optimizers.RMSprop(lr=0.001, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/' + model_name, histogram_freq=0,
                                         write_graph=True, write_images=True)

history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=32, verbose=1,callbacks = [tbCallBack])

model.save(model_name)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", scores)

""" Visualisation """




"""
Next:
Test GRU
Test RecNN
Test Standardisation
Test +ANN
Test + z
"""