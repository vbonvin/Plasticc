import util
import pandas as pd
'''

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

'''
from keras.models import load_model
from keras.models import Sequential
from sklearn.utils import shuffle
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
from sklearn.preprocessing import QuantileTransformer
from keras import regularizers
#from sklearn.preprocessing import PowerTransformer



filter_number1=50

filter_number2=80

filter_size=10



#from keras.callbacks import TensorBoard
# Load Data
init_data = util.readpickle('training_samples.pkl')
data_start = init_data.loc[700:].copy(deep=True)
data_middle=init_data.loc[701:1000].copy(deep=True)
test = init_data.loc[:1000].copy(deep=True)

# Dataset augmentation
#data=data_start
data = dataset_augmentation(data_start, bootstrapping=5)
data_2=dataset_augmentation(data_middle, bootstrapping=2)
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


#data = data.append(epurate_sample(data), ignore_index=True)
'''
print(data.loc[:].values.shape,'shapefinal')

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
[zp_data2, labels2] = dataset_zeropadding(data_2)
[x_test, y_test] = dataset_zeropadding(test)





# Playing around with normalisation -> works great http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
scaler = QuantileTransformer().fit(zp_data)
joblib.dump(scaler, 'QuantileTransformer2.pkl')

zp_data = scaler.transform(zp_data)
zp_data2 = scaler.transform(zp_data2)
x_test = scaler.transform(x_test)

x_train = zp_data
y_train = labels

x_train2 = zp_data2
y_train2 = labels2


y_train = keras.utils.to_categorical(y_train, num_classes=15)
y_train2 = keras.utils.to_categorical(y_train2, num_classes=15)
y_test = keras.utils.to_categorical(y_test, num_classes=15)

print np.shape(y_train)

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
'''
model_m = Sequential()
model_m.add(Reshape((len(zp_data[0])/15., 15), input_shape=(len(zp_data[0]),)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(15, activation='softmax'))
print(model_m.summary())
'''

number_param=22.
model_m = Sequential()
model_m.add(Reshape((len(zp_data[0])/number_param, number_param), input_shape=(len(zp_data[0]),)))
model_m.add(Conv1D(filter_number1, filter_size, activation='relu'))
model_m.add(Conv1D(filter_number1, filter_size, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Dropout(0.5))
model_m.add(Conv1D(filter_number2, filter_size, activation='relu'))
model_m.add(Conv1D(filter_number2, filter_size, activation='relu'))
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



#history = model_m.fit(x_train,y_train,batch_size=32, epochs=30,validation_data=(x_test,y_test),verbose=1)
# Plot training & validation accuracy values

#model_m.save('vanilla_withcoord.h5')

model_m=load_model('vanilla_withcoord.h5')
score = model_m.evaluate(x_test, y_test, batch_size=10)
print("Score: ", score)

print 'evalutaion',model_m.predict(x_train2)
y_prob = model_m.predict(x_train2)
print np.shape(y_prob)
y_classes = y_prob.argmax(axis=-1)

print y_prob
print np.shape(y_train2),np.shape(y_prob)
print 'difference',np.shape(y_train2-y_prob)
indices = [i for i,v in enumerate(y_prob) if np.argmax(y_prob[i,:])!=np.argmax(y_train2[i,:])]
subset_of_wrongly_predicted = [x_train2[i] for i in indices ]
labelsofwrong=[y_train2[i] for i in indices ]

################boost2
y_prob2 = model_m.predict(x_train)
y_classes2 = y_prob.argmax(axis=-1)
indices = [i for i,v in enumerate(y_prob2) if np.argmax(y_prob2[i,:])!=np.argmax(y_train[i,:])]
subset_of_wrongly_predicted2 = [x_train[i] for i in indices ]
labelsofwrong2=[y_train[i] for i in indices ]
#x_train2=np.concatenate((x_train,subset_of_wrongly_predicted2,subset_of_wrongly_predicted,subset_of_wrongly_predicted2,subset_of_wrongly_predicted,subset_of_wrongly_predicted2,subset_of_wrongly_predicted))
#y_train2=np.concatenate((y_train,labelsofwrong2,labelsofwrong,labelsofwrong2,labelsofwrong,labelsofwrong2,labelsofwrong))



filter=np.zeros(len(zp_data[0]))


##boost1
print np.shape(subset_of_wrongly_predicted),np.shape(x_train2)
x_train2=np.concatenate((x_train,subset_of_wrongly_predicted,subset_of_wrongly_predicted,subset_of_wrongly_predicted,subset_of_wrongly_predicted,subset_of_wrongly_predicted,subset_of_wrongly_predicted))
y_train2=np.concatenate((y_train,labelsofwrong,labelsofwrong,labelsofwrong,labelsofwrong,labelsofwrong,labelsofwrong))
print np.shape(x_train2)
#x_train2=np.concatenate((x_train,subset_of_wrongly_predicted))
#y_train2=np.concatenate((y_train,labelsofwrong))
#naive boosting:
#x_train2,ytrain2=shuffle(x_train2,y_train2)
#x_train2,ytrain2=shuffle(x_train2,y_train2)

model_2 = Sequential()
model_2.add(Reshape((len(zp_data[0])/number_param, number_param), input_shape=(len(zp_data[0]),)))
model_2.add(Conv1D(filter_number1, filter_size, activation='relu'))
model_2.add(Conv1D(filter_number1, filter_size, activation='relu'))
model_2.add(MaxPooling1D(3))
model_2.add(Dropout(0.5))
model_2.add(Conv1D(filter_number2, filter_size, activation='relu'))
model_2.add(Conv1D(filter_number2, filter_size, activation='relu'))
model_2.add(GlobalAveragePooling1D())
model_2.add(Dropout(0.5))
model_2.add(Dense(15, activation='softmax'))
print(model_2.summary())


model_2.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

slowadam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model_m.compile(loss='categorical_crossentropy',optimizer=slowadam, metrics=['accuracy'])
#history2 = model_2.fit(x_train2,y_train2,batch_size=32, epochs=30,validation_data=(x_test,y_test),verbose=1)
history2 = model_m.fit(x_train2,y_train2,batch_size=32, epochs=30,validation_data=(x_test,y_test),verbose=1)

#model_2.save('emodel.h5')
#score = model_2.evaluate(x_test, y_test, batch_size=10)
model_m.save('boostwithcoord.h5')
score = model_m.evaluate(x_test, y_test, batch_size=10)
print("Score: ", score)

accuracy=history2.history['acc']
valacc=history2.history['val_acc']
np.save('accuracytwoboost.npy', accuracy)
np.save('valacctwoboost.npy', accuracy)


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