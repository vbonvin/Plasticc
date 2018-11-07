import util
import pandas as pd
'''

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

'''

import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Activation, Input, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from DataProcessing import dataset_handling
from util import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import PowerTransformer

#from keras.callbacks import TensorBoard

init_data = util.readpickle('training_samples.pkl')

data = init_data.copy(deep=True)
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

[zp_data,labels] = dataset_handling(data)
zp_data = MinMaxScaler().fit_transform(zp_data)
#Because of the sofmax or sigmoid activation at the end of the model we have to find a good way to normalize the input between 0 and 1
#Here it is done in the padding in dataprocessing.py, but not ok with t
X = zp_data
Y = labels


##Convert labels to categorical one-hot encoding
Y = keras.utils.to_categorical(Y, num_classes=15)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

"""Network Architecture"""

#vanilla autoencoder

print np.shape(x_train)
print np.amax(x_train),np.amin(x_train)

# this is the size of the encoded representations
encoding_dim = 200  #
intermediate_dim_1=1000
intermediate_dim_2=500

# this is the input vector
input_vector = Input(shape=(len(x_train[0,:]),))

#intermediate layers
x1=Dense(intermediate_dim_1, activation='relu')(input_vector)

xnorm=BatchNormalization()(x1)
xdrop=Dropout(0.5)(xnorm)
x2=Dense(intermediate_dim_2, activation='relu')(xdrop)

# "encoded" is the representation of the input in the latent space
encoded = Dense(encoding_dim, activation='relu')(x2)

#intermediate layers
x3=Dense(intermediate_dim_2, activation='relu')(encoded)
xnorm2=BatchNormalization()(x3)
xdrop2=Dropout(0.5)(xnorm2)

x4=Dense(intermediate_dim_1, activation='relu')(xdrop2)
# "decoded_vector" is reconstruction of the input
decoded_vector = Dense(len(x_train[0,:]), activation='sigmoid')(x3)


# this model maps an input to its reconstruction
autoencoder = Model(input_vector, decoded_vector)


# this model maps an input to its encoded representation
encoder = Model(input_vector, encoded)

#model = load_model('vanilla_autoencoder.h5')

"""Network Training"""

ada = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.01) #decay lr *= (1. / (1. + self.decay * self.iterations))
rmsprop = keras.optimizers.RMSprop(lr=0.001, decay=0.01)
autoencoder.summary()

autoencoder.compile(optimizer=ada,
              loss='binary_crossentropy',
              metrics=['mse'])

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
          #write_graph=True, write_images=True)
#tensorboard("logs/run_a")
history = autoencoder.fit(x_train, x_train, epochs=2, validation_data=(x_test, x_test), batch_size=64, verbose=1)

autoencoder.save('vanilla_autoencoder.h5')

x_decoded = autoencoder.predict(x_train[0:1,:])
print x_decoded
print x_train[0:1,:]


for layer in autoencoder.layers:
    layer.trainable = False
last1=Dense(60, activation='relu')(encoded)
last2=BatchNormalization()(last1)
last3=Dense(15, activation='softmax')(last2)


classifier= Model(input_vector, last3)
classifier.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print np.shape(y_test),np.shape(y_train)
history2 = classifier.fit(x_train, y_train, epochs=5,  validation_data=(x_test, y_test), batch_size=32, verbose=1)


#score = model.evaluate(x_test, y_test, batch_size=100)

"""Network Evaluation"""

#plot_model(autoencoder, to_file='model.png')

#print(score)

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

plt.figure(2)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



plt.figure(3)
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


plt.figure(4)
# Plot training & validation loss values
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
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