import util, sys
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from DataProcessing import *
from ResultAnalysis import predict_lightcurves
from util import *
from tensorflow.keras.models import load_model
from keras.callbacks import TensorBoard
import h5py
from sklearn.externals import joblib
from sklearn.preprocessing import QuantileTransformer
from keras import regularizers
from keras import constraints
from sklearn.model_selection import StratifiedKFold

#model_name = "ANN_P0_Feat960_240_60_B5E5"
#model_name = "ANN_P0_Feat960_480_240_120_60_B5E5"
#model_name = "ANN_P0_FeatX2_960_240_60_B5E5"

#model_name = "ANN_P0_FeatX2_960_240_60_B10E0"
#model_name = "ANN_P0_FeatX2_960_240_60_B5E5_D08"
#model_name = "ANN_P0_FeatX4_960_240_60_B5E5"

#model_name = "ANN_P20_Feat960_240_60_B5E5"



class Generator_ANN(keras.utils.Sequence):

    def __init__(self, data, labels, batch_size):
        self.data, self.labels = data, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.data) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
                for file_name in batch_x]), np.array(batch_y)



if __name__ == "__main__":
    #Load Data
    init_data = util.readpickle('../training_samples_astest.pkl')
    data_start = init_data.loc[1000:].copy(deep=True)
    test = init_data.loc[:1000].copy(deep=True)

    #

    model_name = "ANNX2_P0_3Layers_B0E0_V3000_KL001_BL001_None_Kfold2"

    kernel_regularizer = 0.001
    kernel_constraint = None

    #Dataset augmentation

    # We will use the spl parameters, so no augmentation for the moment (would require recomputing new fitparams after bootstraping, takes a long time...)
    data = dataset_augmentation(data_start, bootstrapping=5, epurate=5)

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

    [zp_data,labels] = dataset_zeropadding(data)
    [x_test,y_test] = dataset_zeropadding(test)

    #Playing around with normalisation -> works great http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    scaler = QuantileTransformer(output_distribution='uniform').fit(zp_data)
    joblib.dump(scaler, model_name + '_scaler.pkl')

    zp_data = scaler.transform(zp_data)
    x_test = scaler.transform(x_test)

    x_train = zp_data
    y_train = labels

    y_test = keras.utils.to_categorical(y_test, num_classes=15)

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
        # Zeropadded ANN
        mult_factor = 4

        model = keras.Sequential([
            keras.layers.Dense(960 * mult_factor, input_shape=(len(zp_data[0]),), activation=tf.nn.elu,
                               kernel_regularizer=regularizers.l2(kernel_regularizer),
                               bias_regularizer=regularizers.l2(kernel_regularizer),
                               kernel_constraint=kernel_constraint,
                               bias_constraint=kernel_constraint),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dropout(0.5),
            #keras.layers.Dense(480 * mult_factor, activation=tf.nn.elu,
            #                   kernel_regularizer=regularizers.l2(kernel_regularizer),
            #                   bias_regularizer=regularizers.l2(kernel_regularizer),
            #                   kernel_constraint=kernel_constraint,
            #                   bias_constraint=kernel_constraint),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(240 * mult_factor, activation=tf.nn.elu,
                               kernel_regularizer=regularizers.l2(kernel_regularizer),
                               bias_regularizer=regularizers.l2(kernel_regularizer),
                               kernel_constraint=kernel_constraint,
                               bias_constraint=kernel_constraint),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dropout(0.5),
            #keras.layers.Dense(120 * mult_factor, activation=tf.nn.elu,
            #                   kernel_regularizer=regularizers.l2(kernel_regularizer),
            #                   bias_regularizer=regularizers.l2(kernel_regularizer),
            #                   kernel_constraint=kernel_constraint,
            #                   bias_constraint=kernel_constraint),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(60 * mult_factor, activation=tf.nn.elu,
                               kernel_regularizer=regularizers.l2(kernel_regularizer),
                               bias_regularizer=regularizers.l2(kernel_regularizer),
                               kernel_constraint=kernel_constraint,
                               bias_constraint=kernel_constraint),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(15, activation=tf.nn.softmax)
        ])

        # Compile model
        ada = keras.optimizers.Adagrad(lr=0.01, epsilon=None,
                                       decay=0.01)  # decay lr *= (1. / (1. + self.decay * self.iterations))
        rmsprop = keras.optimizers.RMSprop(lr=0.001, decay=0.01)
        model.compile(optimizer=ada,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Fit the model
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/' + model_name, histogram_freq=0,
                                                 write_graph=True, write_images=True)
        modelcheckpointCallBack = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',
                                                                  save_weights_only=True, period=10)

        ##Convert labels to categorical one-hot encoding
        y_train_hotcoded = keras.utils.to_categorical(y_train[train], num_classes=15)
        y_test_hotcoded = keras.utils.to_categorical(y_train[test], num_classes=15)

        history = model.fit(x_train[train], y_train_hotcoded, epochs=50, validation_data=(x_test, y_test), batch_size=512, verbose=1,
                            callbacks=[tbCallBack, modelcheckpointCallBack])
        # evaluate the model
        scores = model.evaluate(x_train[test], y_test_hotcoded, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))







    #del model  # deletes the existing model

    #model = load_model('ANN_3layers_bootstrap5.h5')

    """Network Training"""
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + "_model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_name + "_model_weights.h5")
    """



    """Network Evaluation"""

    score = model.evaluate(x_test, y_test, batch_size=10)
    print("Score: ", score)

    y_test = predict_lightcurves(model,x_test,y_test)
    print(y_test)
    plt.show()
