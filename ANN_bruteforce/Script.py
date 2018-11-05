import util
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from DataProcessing import dataset_handling
from util import *
from tensorflow.keras.models import load_model
from keras.callbacks import TensorBoard

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

init_data = util.readpickle('training_samples.pkl')

data = init_data.copy(deep=True)
"""
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
"""
#data = data.append(epurate_sample(data), ignore_index=True)

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
[zp_data,labels] = dataset_handling(data)
#Playing around with normalisation -> works great http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
#zp_data = tf.keras.utils.normalize(zp_data,axis=0)
zp_data = StandardScaler().fit_transform(zp_data)
#zp_data = QuantileTransformer(output_distribution='uniform').fit_transform(zp_data)
#zp_data = MinMaxScaler().fit_transform(zp_data)
#zp_data = QuantileTransformer(output_distribution='normal').fit_transform(zp_data)
x_train = zp_data
y_train = labels

##Convert labels to categorical one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=15)

"""Network Architecture"""

#Zeropadded ANN

model = keras.Sequential([
    keras.layers.Dense(960, input_shape=(len(zp_data[0]),), activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    #keras.layers.Dense(480, activation=tf.nn.relu),
    #keras.layers.BatchNormalization(),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    #keras.layers.Dense(120, activation=tf.nn.relu),
    #keras.layers.BatchNormalization(),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(60, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(15, activation=tf.nn.softmax)
])

#del model  # deletes the existing model

#model = load_model('ANN_3layers_model.h5')

"""Network Training"""

ada = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.01) #decay lr *= (1. / (1. + self.decay * self.iterations))
rmsprop = keras.optimizers.RMSprop(lr=0.001, decay=0.01)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/run_StandardScaler', histogram_freq=0,
          write_graph=True, write_images=True)
#tensorboard("logs/run_a")
history = model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32, verbose=1,callbacks = [tbCallBack])

model.save('ANN_3layers_model_normalised.h5')

#score = model.evaluate(x_test, y_test, batch_size=100)

"""Network Evaluation"""

plot_model(model, to_file='model.png')

#print(score)

# Plot training & validation accuracy values
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


plt.figure(2)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Train the model, iterating on the data in batches of 32 samples
#for i in range(10):
#    print("Starting Epoch %d" %i)
#    model.fit(x_train, y_train, epochs=10, batch_size=32)
#    score = model.evaluate(x_test, y_test, batch_size=128)
#    print("Accuracy %f", score)



#Zeropadded ANN