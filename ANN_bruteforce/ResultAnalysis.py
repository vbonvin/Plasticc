import util
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
import itertools
from DataProcessing import *
from util import *
from tensorflow.keras.models import load_model
from keras.callbacks import TensorBoard
import h5py
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Credit: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def predict_lightcurves(model,x_test,y_test):
    """
    This function returns the predicted values, plots confusion matrix and more.
    Mostly a summarize function for Christoph
    """

    #predict data
    y_pred = model.predict(x_test)
    print(y_pred,y_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    np.set_printoptions(precision=2)

    # Define class_names
    class_names = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99])

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')

    plt.show()
    return y_pred

if __name__ == "__main__":
    #Loading ANN model
    model = load_model('../ANN_3layers_model_standardised.h5')
    #model.compile(optimizer=rmsprop,loss='categorical_crossentropy', metrics=['accuracy'])

    #Loading and preparing Test data
    init_data = util.readpickle('../training_samples.pkl')
    #data = init_data.loc[1000:].copy(deep=True)
    test_data = init_data.loc[:1000].copy(deep=True)
    [x_test,y_test] = dataset_zeropadding(test_data)

    scaler = joblib.load('../QuantileTransformer.pkl')
    #scaler = QuantileTransformer(QuantileTransformer(copy=True, ignore_implicit_zeros=False, n_quantiles=1000,
    #          output_distribution='uniform', random_state=None,
    #         subsample=100000))
    x_test = scaler.transform(x_test)
    #y_test = scaler.transform(y_test)
    #scaler.set_params()


    y_test = keras.utils.to_categorical(y_test, num_classes=15)

    score = model.evaluate(x_test, y_test, batch_size=100)
    print("Score: ", score)


    y_test = predict_lightcurves(model,x_test,y_test)
    print(y_test)

