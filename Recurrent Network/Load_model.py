import util
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from util import *
from sklearn.externals import joblib
from keras.models import model_from_json

from ANN_bruteforce.DataProcessing import *
from ANN_bruteforce.ResultAnalysis import predict_lightcurves
from Chris_ML_lib import compute_weights, w_categorical_crossentropy

"Taken from https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/"

# Recurrent Network
model_name = "LSTM_P0_Feat50_B0E0"

"""  Load Data """
init_data = util.readpickle('../training_samples_astest.pkl')
data_start = init_data.loc[1000:].copy(deep=True)
test = init_data.loc[:1000].copy(deep=True)

""" Data Preparation """

[x_test, y_test] = dataset_zeropadding_3D(test)

scaler = joblib.load(model_name + '_scaler.pkl')
for ii in range(len(scaler)):
    x_test[:, ii, :] = scaler[ii].transform(x_test[:, ii, :])

y_test = keras.utils.to_categorical(y_test, num_classes=15)

## Creating Weights: the critical and arbitrary factor is the penalty_factor
N_of_Classes = np.array([151., 495., 924., 1193., 183., 30., 484., 102., 981.,  208., 370.,  2313.,  239., 175.])
penalty_factor = 20

w_array = compute_weights(N_of_Classes, penalty_factor=20)

#loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)

""" Model """

# load json and create model
json_file = open(model_name + '_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_name + "_model_weights.h5")
print("Loaded model from disk")


loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)
loaded_model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])

""" Visualisation """

score = loaded_model.evaluate(x_test, y_test, batch_size=10)
print("Score: ", score)

y_test = predict_lightcurves(loaded_model, x_test, y_test)
print(y_test)
plt.show()
