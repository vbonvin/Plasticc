import util
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np


data = util.readpickle('training_samples.pkl')

print(data)
print(type(data.iloc[0]))
print(data.axes)
#print(data.loc[:, u'fluxes_0'])

"""Dateset Handling"""
#Flatten Data into 1D Vector
#Beginning just fluxes and time data

#First Problematic: Variable input
# -> zeropadded ANN
# -> Recurrent neural network https://en.wikipedia.org/wiki/Recurrent_neural_network
# -> Recursive neural network
##Zeropadded ANN
###Maximum number of points = 72 , keep around 80 values for even number
###
#max_len = np.max([len(a) for a in arr])
max_len = 80
zp_data = data.loc[ :,[u'fluxes_0',u'fluxes_1',u'fluxes_2',u'fluxes_3',u'fluxes_4',u'fluxes_5',u'mjds_0',u'mjds_1',u'mjds_2',u'mjds_3',u'mjds_4',u'mjds_0']].values
###Zero-padding using Numpy
zp_data = np.asarray([[np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in data] for data in zp_data])

print(zp_data[0])



"""Network Architecture"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


"""Network Training"""



#####Archive
"""
###Maximum number of points = 72 , keep around 80 values for even number
zp_data = data.loc[:, u'fluxes_0'].values
N_max = 0
for i in range(len(zp_data)):
    if N_max < len(zp_data[i]):
        N_max = len(zp_data[i])
        print(N_max)
print(N_max)





"""