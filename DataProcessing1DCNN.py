import util
import pandas as pd
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from util import *
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
#init_data = util.readpickle('training_samples.pkl')

#data = init_data.copy(deep=True)
#print(data)
#data = data.append(bootstrap_sample(init_data), ignore_index=True)
#print(data.loc[7850])
#data = data.append(bootstrap_sample(init_data))
#data = data.append(bootstrap_sample(init_data))
#data = data.append(bootstrap_sample(init_data))
#data = data.append(bootstrap_sample(init_data))
#data = data.append(bootstrap_sample(init_data))
#data = data.append(bootstrap_sample(init_data))
#data = data.append(bootstrap_sample(init_data))
#data = data.append(epurate_sample(data), ignore_index=True)

#print(data)
#data = epurate_sample(data)
#print(data)
#print(data.loc[:].values.shape)


def dataset_augmentation(data_start, bootstrapping = 1, shuffle = True):
    """
    Returns augmented dataset using bootstrap and epurate fonctions from utils
    """
    data = data_start
    for ii in range(bootstrapping):
        data = data.append(bootstrap_sample(data_start), ignore_index=True)

    # Shuffling (Important)
    if shuffle == True:
        data = data.sample(frac=1)
    return data

def dataset_zeropadding(data):
    """
    Returns zeropadded and flattened dataset and labels. Essently converts panda data into numpy data and formats for keras.
    This is done with standardisation.
    """
    # Flatten Data into 1D Vector
    ##Maximum number of points = 72 , keep around 80 values for even number
    ####max_len = np.max([len(a) for a in arr])
    max_len = 80
    '''
    zp_data = data.loc[ :,[u'fluxes_0',
                           u'fluxes_1',
                           u'fluxes_2', u'fluxes_3', u'fluxes_4', u'fluxes_5']].values
    #                       u'fluxerrs_1',u'fluxerrs_2',u'fluxerrs_3',u'fluxerrs_4',u'fluxerrs_5',u'fluxes_0',u'fluxes_1',
    #                       u'fluxes_2',u'fluxes_3',u'fluxes_4',u'fluxes_5',u'mjds_0',u'mjds_1',u'mjds_2',u'mjds_3',u'mjds_4',
    #                       u'mjds_5']].values
    #
    '''
    zp_data = data.loc[:, [u'fluxes_0',
                           u'fluxes_1',
                           u'fluxes_2', u'fluxes_3', u'fluxes_4', u'fluxes_5',u'fluxerrs_0',u'fluxerrs_1',u'fluxerrs_2',u'fluxerrs_3',u'fluxerrs_4',u'fluxerrs_5',u'mjds_0',u'mjds_1',u'mjds_2',u'mjds_3',u'mjds_4',
                           u'mjds_5']].values



    ##Zero-padding using Numpy and reshape in 1d vector [:,data]
    zp_data = np.asarray(
        [[np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in item] for item in zp_data])
    zp_data = zp_data.reshape(zp_data.shape[0], -1)
    print np.shape(zp_data),'shape1'
    #zp_data = np.c_[
        #zp_data, data.loc[:, [u'gal_b', u'gal_l', u'hostgal_photoz', u'hostgal_photoz_err', u'hostgal_specz']].values]
    ##Normalise data to be determined


    z=np.repeat(data.loc[:, [u'hostgal_photoz']].values,max_len)

    z=z.reshape(zp_data.shape[0], -1)


    zp_data=np.c_[zp_data,z]

    zerr = np.repeat(data.loc[:, [u'hostgal_photoz_err']].values, max_len)
    zerr = zerr.reshape(zp_data.shape[0], -1)

    zp_data=np.c_[zp_data, zerr]
    '''
    ugalb = np.repeat(data.loc[:, [u'gal_b']].values, max_len)
    ugalb = ugalb.reshape(zp_data.shape[0], -1)

    zp_data = np.c_[zp_data, ugalb]

    ugall = np.repeat(data.loc[:, [u'gal_l']].values, max_len)
    ugall = ugall.reshape(zp_data.shape[0], -1)

    zp_data = np.c_[zp_data, ugall]
    '''


    ##Load labels and convert to integer
    labels = data.loc[:, [u'target']].values
    labels = labels.flatten()
    labels_name = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99])
    [np.place(labels, labels == labels_name[i], [i]) for i in range(len(labels_name))]
    #print(zp_data.shape)

    return[zp_data,labels]

def dataset_handling_with_standardisation(init_data):
    """
    Returns standardised, zeropadded and flattened dataset and labels. Essently converts panda data into numpy data and formats for keras.
    This is done with standardisation.
    """
    #
    ##Maximum number of points = 72 , keep around 80 values for even number
    max_len = 80
    ##Fluxes, Standardisation is done over 1 type of feature
    data = init_data.loc[:, [u'fluxes_0', u'fluxes_1', u'fluxes_2', u'fluxes_3', u'fluxes_4', u'fluxes_5']].values
    zp_array_flux = []
    for dat in data:
        n_data = []
        for ii in range(len(dat)):
            n_data = np.append(n_data, np.pad(dat[ii], (0, max_len * 5 - len(dat[ii])), 'constant', constant_values=0))
        n_data = MinMaxScaler().fit_transform(n_data.reshape(-1, 1)).flatten()
        zp_array_flux.append(n_data)
        #print'erraefjawie', n_data
    zp_array_flux = np.array(zp_array_flux)
    print(zp_array_flux.shape)

    ##Fluxerrors, Standardisation is done over 1 type of feature
    data = init_data.loc[:,
           [u'fluxerrs_0', u'fluxerrs_1', u'fluxerrs_2', u'fluxerrs_3', u'fluxerrs_4', u'fluxerrs_5']].values
    zp_array_flux_error = []
    for dat in data:
        n_data = []
        for ii in range(len(dat)):
            n_data = np.append(n_data, np.pad(dat[ii], (0, max_len * 5 - len(dat[ii])), 'constant', constant_values=0))
        n_data = MinMaxScaler().fit_transform(n_data.reshape(-1, 1)).flatten()
        zp_array_flux_error.append(n_data)

    zp_array_flux_error = np.array(zp_array_flux_error)
    print(zp_array_flux_error.shape)

    ##Time, Standardisation is done over 1 type of feature
    data = init_data.loc[:, [u'mjds_0', u'mjds_1', u'mjds_2', u'mjds_3', u'mjds_4', u'mjds_5']].values
    zp_array_mjds = []
    for dat in data:
        n_data = []
        for ii in range(len(dat)):
            n_data = np.append(n_data, np.pad(dat[ii], (0, max_len * 5 - len(dat[ii])), 'constant', constant_values=0))
        n_data = MinMaxScaler().fit_transform(n_data.reshape(-1, 1)).flatten()

        zp_array_mjds.append(n_data)
    zp_array_mjds = np.array(zp_array_mjds)
    print(zp_array_mjds.shape)

    ##Concatenating everything
    zp_data = np.c_[zp_array_flux, zp_array_flux_error, zp_array_mjds]

    ##Adding redshift info// Gal pos info might be necessary to remove
    datared=init_data.loc[:, [u'gal_b', u'gal_l', u'hostgal_photoz', u'hostgal_photoz_err', u'hostgal_specz']].values
    datared=MinMaxScaler().fit_transform(datared)
    print np.shape(datared)
    zp_data = np.c_[
        zp_data, datared]
    print(zp_data.shape)

    ##Load labels and convert to integer
    labels = init_data.loc[:, [u'target']].values
    labels = labels.flatten()
    labels_name = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99])
    [np.place(labels, labels == labels_name[i], [i]) for i in range(len(labels_name))]

    return [zp_data, labels]

#dataset_handling(data)

