import util
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from DataProcessing import *
from util import *
from tensorflow.keras.models import load_model
from keras.callbacks import TensorBoard
import h5py
from sklearn.externals import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

init_data = util.readpickle('training_samples.pkl')
data_start = init_data.loc[1000:].copy(deep=True)
test = init_data.loc[:1000].copy(deep=True)

data = data_start.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
#10
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
data = data.append(bootstrap_sample(data_start), ignore_index=True)
#20

#Shuffling (Important)
data = data.sample(frac=1)
#data = data.append(epurate_sample(data), ignore_index=True)
print(data.loc[:].values.shape)

"""Dateset Handling"""
#Flatten Data into 1D Vector
#Beginning just fluxes and time data

#First Problematic: Variable input
# -> zeropadded ANN
# -> Recurrent neural network https://en.wikipedia.org/wiki/Recurrent_neural_network
# -> Recursive neural network

#Zeropadded ANN
#[zp_data,labels] = dataset_handling_with_standardisation(data)
#[x_test,y_test] = dataset_handling_with_standardisation(test)

print("Finished loading")
[zp_data,labels] = dataset_handling(data)
[x_test,y_test] = dataset_handling(test)
#Playing around with normalisation -> works great http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
scaler = QuantileTransformer(output_distribution='uniform').fit(zp_data)
joblib.dump(scaler, 'QuantileTransformer.pkl')
print(scaler)
print(scaler.get_params())
print(scaler.get_params())
#print(scaler.fit_params)
zp_data = scaler.transform(zp_data)
x_test = scaler.transform(x_test)
###zp_data = QuantileTransformer(output_distribution='normal').fit_transform(zp_data)
x_train = zp_data
y_train = labels

##Convert labels to categorical one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=15)
y_test = keras.utils.to_categorical(y_test, num_classes=15)

#y_train = np.load('y_train.npy')
#y_test = np.load('y_test.npy')

hf = h5py.File('data.h5', 'w')
hf.create_dataset('x_train', data=x_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('x_test', data=x_test)
hf.create_dataset('y_test', data=y_test)

hf.close()

###np.save('x_train.npy', x_train)
###np.save('y_train.npy', y_train)
###np.save('x_test.npy', x_test)
###np.save('y_test.npy', y_test)




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

#model = load_model('ANN_3layers_bootstrap5.h5')

"""Network Training"""

ada = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.01) #decay lr *= (1. / (1. + self.decay * self.iterations))
rmsprop = keras.optimizers.RMSprop(lr=0.001, decay=0.01)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/run_bootstrap_20_ANN3layers', histogram_freq=0,
          write_graph=True, write_images=True)
#tensorboard("logs/run_a")
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=1,callbacks = [tbCallBack])

model.save('ANN_3layers_model_standardised.h5')

score = model.evaluate(x_test, y_test, batch_size=100)
print("Score: ", score)

"""Network Evaluation"""

plot_model(model, to_file='model.png')

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