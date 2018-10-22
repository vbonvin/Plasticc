import util
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model

data = util.readpickle('training_samples.pkl')

"""Dateset Handling"""
#Flatten Data into 1D Vector
#Beginning just fluxes and time data

#First Problematic: Variable input
# -> zeropadded ANN
# -> Recurrent neural network https://en.wikipedia.org/wiki/Recurrent_neural_network
# -> Recursive neural network

#Zeropadded ANN

##Maximum number of points = 72 , keep around 80 values for even number
####max_len = np.max([len(a) for a in arr])
max_len = 80
zp_data = data.loc[ :,[u'fluxes_0',u'fluxes_1',u'fluxes_2',u'fluxes_3',u'fluxes_4',u'fluxes_5',u'mjds_0',u'mjds_1',u'mjds_2',u'mjds_3',u'mjds_4',u'mjds_5']].values
##Zero-padding using Numpy and reshape in 1d vector [:,data]
zp_data = np.asarray([[np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in item] for item in zp_data])
zp_data = zp_data.reshape(zp_data.shape[0],-1)

##Load labels and convert to integer
labels = data.loc[ :,[u'target']].values
labels = labels.flatten()
labels_name = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99])
[np.place(labels, labels == labels_name[i],[i]) for i in range(len(labels_name))]

##Dividing into train and validation
x_train = zp_data[:len(zp_data)-100]
y_train = labels[:len(zp_data)-100]
x_test = zp_data[len(zp_data)-100:len(zp_data)]
y_test = labels[len(zp_data)-100:len(zp_data)]

##Convert labels to categorical one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=15)
y_test = keras.utils.to_categorical(y_test, num_classes=15)

"""Network Architecture"""

#Zeropadded ANN

model = keras.Sequential([
    keras.layers.Dense(480, input_shape=(960,), activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(240, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(60, activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(15, activation=tf.nn.softmax)
])


"""Network Training"""

ada = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.01) #decay lr *= (1. / (1. + self.decay * self.iterations))
rmsprop = keras.optimizers.RMSprop(lr=0.001, decay=0.01)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=32, verbose=1)

score = model.evaluate(x_test, y_test, batch_size=100)

"""Network Evaluation"""

plot_model(model, to_file='model.png')

print(score)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

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