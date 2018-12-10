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

#model_name = "ANN_P0_Feat960_240_60_B5E5"
#model_name = "ANN_P0_Feat960_480_240_120_60_B5E5"
#model_name = "ANN_P0_FeatX2_960_240_60_B5E5"

#model_name = "ANN_P0_FeatX2_960_240_60_B10E0"
#model_name = "ANN_P0_FeatX2_960_240_60_B5E5_D08"
#model_name = "ANN_P0_FeatX4_960_240_60_B5E5"

#model_name = "ANN_P20_Feat960_240_60_B5E5"

if __name__ == "__main__":
    #Load Data
    init_data = util.readpickle('../training_samples_astest.pkl')
    data_start = init_data.loc[1000:].copy(deep=True)
    test = init_data.loc[:1000].copy(deep=True)

    model_name = "ANN_P0_FeatX2_960_240_60_B5E5_D08"

    #Dataset augmentation

    # We will use the spl parameters, so no augmentation for the moment (would require recomputing new fitparams after bootstraping, takes a long time...)
    data = dataset_augmentation(data_start, bootstrapping=10, epurate=0)

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

    ##Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=15)
    y_test = keras.utils.to_categorical(y_test, num_classes=15)

    #y_train = np.load('y_train.npy')
    #y_test = np.load('y_test.npy')

    #hf = h5py.File('data.h5', 'w')
    #hf.create_dataset('x_train', data=x_train)
    #hf.create_dataset('y_train', data=y_train)
    #hf.create_dataset('x_test', data=x_test)
    #hf.create_dataset('y_test', data=y_test)

    #hf.close()

    ###np.save('x_train.npy', x_train)
    ###np.save('y_train.npy', y_train)
    ###np.save('x_test.npy', x_test)
    ###np.save('y_test.npy', y_test)




    """Network Architecture"""

    #Zeropadded ANN
    mult_factor = 4

    model = keras.Sequential([
        keras.layers.Dense(960*mult_factor, input_shape=(len(zp_data[0]),), activation=tf.nn.elu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        #keras.layers.Dense(480*2, activation=tf.nn.relu),
        #keras.layers.BatchNormalization(),
        #keras.layers.Dropout(0.5),
        keras.layers.Dense(240*mult_factor, activation=tf.nn.elu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        #keras.layers.Dense(120*2, activation=tf.nn.relu),
        #keras.layers.BatchNormalization(),
        #keras.layers.Dropout(0.5),
        keras.layers.Dense(60*mult_factor, activation=tf.nn.elu),
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

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/' + model_name, histogram_freq=0,
              write_graph=True, write_images=True)
    modelcheckpointCallBack = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',
                                                              save_weights_only=True, period=10)

    history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=32, verbose=1,callbacks = [tbCallBack,modelcheckpointCallBack])

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
