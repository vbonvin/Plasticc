from keras.models import load_model
from keras.models import Sequential
from sklearn.utils import shuffle
import keras
import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling2D, Dense,GlobalAveragePooling1D, BatchNormalization, Activation, Input, Flatten, Dropout, Reshape,MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from DataProcessing import *
from util import *
import matplotlib.pyplot as plt
from keras.utils import plot_model
import itertools
from DataProcessing import *
from util import *
from keras.models import load_model
from keras.callbacks import TensorBoard
import h5py
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import keras.backend as K
from itertools import product
from functools import partial
import sys

picklepath=sys.argv[1]
scalerpath=sys.argv[2]
modelpath=sys.argv[3]
identificationpacket=sys.argv[4]

#picklepath='./training_samples.pkl'
#scalerpath='./QuantileTransformer2.pkl'
#modelpath='./boostnwelossinvfilter.h5'

""" Loss function definition """
##Personalised loss function , example taken from https://github.com/keras-team/keras/issues/2115
##Test needed comparing categorical cross entropy to weigthed cat cross entropy (without weights)
def compute_weights(N_of_Classes, penalty_factor=20):
    "Creating Weights: the critical and arbitrary factor is the penalty_factor"
    w_array = np.ones((len(N_of_Classes)+1, len(N_of_Classes)+1))

    for c_p, c_t in product(range(len(w_array)-1), range(len(w_array)-1)):
        w_array[c_t, c_p] = 1 + N_of_Classes[c_p]/np.sum(N_of_Classes) * penalty_factor        #Penalising 1 False classified as 0 -> has to be higher than 1
    return w_array

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
penalty_factor = 20

w_array = compute_weights(N_of_Classes, penalty_factor=20)

loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)

#"loading the data"

# Loading and preparing Test data
init_data = util.readpickle(picklepath)

test_data = init_data.loc[:].copy(deep=True)

[x_test, y_test,id] = dataset_zeropadding(test_data,idnumber=True)
scaler = joblib.load(scalerpath)
#"loading the model"
print 'id',id
filter_number1 = 50

filter_number2 = 80
filter_number3 = 50

filter_size = 10
number_param = 20.
model = Sequential()
model.add(Reshape((len(x_test[0]) / number_param, number_param), input_shape=(len(x_test[0]),)))
model.add(Conv1D(filter_number1, filter_size, activation='relu'))
model.add(Conv1D(filter_number1, filter_size, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Conv1D(filter_number2, filter_size, activation='relu'))
model.add(Conv1D(filter_number2, filter_size, activation='relu'))

model.add(GlobalAveragePooling1D())

model.add(Dropout(0.5))

model.add(Dense(15, activation='softmax'))
print(model.summary())
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

x_test = scaler.transform(x_test)

#load the weights
model.load_weights(modelpath)


#prediction:
prediction = model.predict(x_test)

print prediction

