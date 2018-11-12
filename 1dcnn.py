import util
import pandas as pd
'''

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

'''
from keras.models import Sequential
import keras
from keras.models import Model
from keras.layers import Conv1D, MaxPooling2D, Dense,GlobalAveragePooling1D, BatchNormalization, Activation, Input, Flatten, Dropout, Reshape,MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from DataProcessing import *
from util import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib
#from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import PowerTransformer

#from keras.callbacks import TensorBoard
# Load Data
init_data = util.readpickle('training_samples.pkl')
data_start = init_data.loc[1000:].copy(deep=True)
test = init_data.loc[:1000].copy(deep=True)

# Dataset augmentation
#data=data_start
data = dataset_augmentation(data_start, bootstrapping=5)
# data = data.append(epurate_sample(data), ignore_index=True)

'''
data = data.append(bootstrap_sample(init_data), ignore_index=True)

data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)

data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)

data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)
data = data.append(bootstrap_sample(init_data), ignore_index=True)

#data = data.append(epurate_sample(data), ignore_index=True)
'''
print(data.loc[:].values.shape)

#print(data.loc[0])

"""Dateset Handling"""
#Flatten Data into 1D Vector
#Beginning just fluxes and time data

#First Problematic: Variable input
# -> zeropadded ANN
# -> Recurrent neural network https://en.wikipedia.org/wiki/Recurrent_neural_network
# -> Recursive neural network

#Zeropadded ANN



[zp_data, labels] = dataset_zeropadding(data)
[x_test, y_test] = dataset_zeropadding(test)

# Playing around with normalisation -> works great http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
scaler = MaxAbsScaler().fit(zp_data)
joblib.dump(scaler, 'MaxAbsScaler.pkl')

zp_data = scaler.transform(zp_data)
x_test = scaler.transform(x_test)

x_train = zp_data
y_train = labels

y_train = keras.utils.to_categorical(y_train, num_classes=15)
y_test = keras.utils.to_categorical(y_test, num_classes=15)

'''
[zp_data,labels] =dataset_handling_with_standardisation(data)
#zp_data = MinMaxScaler().fit_transform(zp_data)
#Because of the sofmax or sigmoid activation at the end of the model we have to find a good way to normalize the input between 0 and 1

X = zp_data.astype(np.float32)
Y = labels


##Convert labels to categorical one-hot encoding
Y = (keras.utils.to_categorical(Y, num_classes=15)).astype(np.float32)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
'''
"""Network Architecture"""

#vanilla 1d cnn
#for the moment just for the fluxes =>1/5 nd 5 channels

model_m = Sequential()
model_m.add(Reshape((len(zp_data[0])/5., 5), input_shape=(len(zp_data[0]),)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(15, activation='softmax'))
print(model_m.summary())
"""Network Evaluation"""



model_m.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
#y_test = predict_lightcurves(model, x_test, y_test)
#print(y_test)

#plot_model(autoencoder, to_file='model.png')

#print(score)
score = model_m.evaluate(x_test, y_test, batch_size=10)
print("Score: ", score)


history = model_m.fit(x_train,y_train,batch_size=32, epochs=20,validation_split=0.2,verbose=1)
# Plot training & validation accuracy values
'''
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
'''
'''
plt.figure()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Autoencoder loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

'''


# Train the model, iterating on the data in batches of 32 samples
#for i in range(10):
#    print("Starting Epoch %d" %i)
#    model.fit(x_train, y_train, epochs=10, batch_size=32)
#    score = model.evaluate(x_test, y_test, batch_size=128)
#    print("Accuracy %f", score)



#Zeropadded ANN