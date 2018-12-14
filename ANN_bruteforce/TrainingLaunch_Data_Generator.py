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

""

class Data_Generator_ANN(keras.utils.Sequence):

    def __init__(self, data, scaler, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.scaler = scaler

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        #Apply random data augmentation
        batch_x = batch_x.apply(bootstrap_sample, axis=1)
        batch_x = batch_x.apply(epurate_sample, axis=1)

        # Zeropadded ANN
        [batch_x, batch_y] = dataset_zeropadding(batch_x)
        batch_x = self.scaler.transform(batch_x)

        batch_y = keras.utils.to_categorical(batch_y, num_classes=15)


        return batch_x, batch_y



if __name__ == "__main__":
    #Load Data
    init_data = util.readpickle('../training_samples_astest.pkl')
    data = init_data.loc[3000:].copy(deep=True)
    test = init_data.loc[:3000].copy(deep=True)

    #

    model_name = "ANNX2_P0_5Layers_B5E5_V3000_KL001_BL001_UnitNorm"

    kernel_regularizer = 0.001
    kernel_constraint = constraints.UnitNorm(axis=0)

    """Dateset Handling"""
    #First Problematic: Variable input
    # -> zeropadded ANN
    # -> Recurrent neural network https://en.wikipedia.org/wiki/Recurrent_neural_network
    # -> Recursive neural network

    #Zeropadded ANN
    [x_train,y_train] = dataset_zeropadding(data)
    [x_test,y_test] = dataset_zeropadding(test)

    #Playing around with normalisation -> works great http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    scaler = QuantileTransformer(output_distribution='uniform').fit(x_train)
    joblib.dump(scaler, model_name + '_scaler.pkl')

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    batch_size = 128
    #declare generator
    train_data_generator = Data_Generator_ANN(data, scaler, batch_size)
    print("Len generator",int(np.ceil(len(train_data_generator.data)/ float(train_data_generator.batch_size))))
    print(train_data_generator.__getitem__(0))

    ##Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=15)
    y_test = keras.utils.to_categorical(y_test, num_classes=15)



    """Network Architecture"""

    #Zeropadded ANN
    mult_factor = 4

    model = keras.Sequential([
        keras.layers.Dense(960*mult_factor, input_shape=(len(x_train[0]),), activation=tf.nn.elu,
                           kernel_regularizer=regularizers.l2(kernel_regularizer),
                           bias_regularizer=regularizers.l2(kernel_regularizer), kernel_constraint = kernel_constraint,
                           bias_constraint = kernel_constraint),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(480*mult_factor, activation=tf.nn.elu, kernel_regularizer=regularizers.l2(kernel_regularizer),
                           bias_regularizer=regularizers.l2(kernel_regularizer), kernel_constraint = kernel_constraint,
                           bias_constraint = kernel_constraint),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(240*mult_factor, activation=tf.nn.elu, kernel_regularizer=regularizers.l2(kernel_regularizer),
                           bias_regularizer=regularizers.l2(kernel_regularizer), kernel_constraint = kernel_constraint,
                           bias_constraint = kernel_constraint),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(120*mult_factor, activation=tf.nn.elu, kernel_regularizer=regularizers.l2(kernel_regularizer),
                           bias_regularizer=regularizers.l2(kernel_regularizer), kernel_constraint = kernel_constraint,
                           bias_constraint = kernel_constraint),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(60*mult_factor, activation=tf.nn.elu, kernel_regularizer=regularizers.l2(kernel_regularizer),
                           bias_regularizer=regularizers.l2(kernel_regularizer), kernel_constraint = kernel_constraint,
                           bias_constraint = kernel_constraint),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(15, activation=tf.nn.softmax)
    ])

    #del model  # deletes the existing model

    #model = load_model('ANN_3layers_bootstrap5.h5')

    """Network Training"""

    ada = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.01) #decay lr *= (1. / (1. + self.decay * self.iterations))
    rmsprop = keras.optimizers.RMSprop(lr=0.001, decay=0.01)
    model.compile(optimizer=ada,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/' + model_name, histogram_freq=0,
              write_graph=True, write_images=True)
    modelcheckpointCallBack = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',
                                                              save_weights_only=True, period=10)

    history = model.fit_generator(generator=train_data_generator, epochs=100, use_multiprocessing=True,
                    workers=8, validation_data=(x_test, y_test), verbose=1, callbacks=[tbCallBack,modelcheckpointCallBack])

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + "_model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_name + "_model_weights.h5")

    """Network Evaluation"""

    score = model.evaluate(x_test, y_test, batch_size=10)
    print("Score: ", score)

    y_test = predict_lightcurves(model,x_test,y_test)
    print(y_test)
    plt.show()
