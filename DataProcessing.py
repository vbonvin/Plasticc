import util
import pandas as pd
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from util import *

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


def dataset_handling(data):
    global zp_data, labels
    """Dateset Handling"""
    # Flatten Data into 1D Vector
    ##Maximum number of points = 72 , keep around 80 values for even number
    ####max_len = np.max([len(a) for a in arr])
    max_len = 80
    # zp_data = data.loc[ :,[u'gal_b',u'gal_l',u'hostgal_photoz',u'hostgal_photoz_err',u'hostgal_specz',u'fluxerrs_0',
    #                       u'fluxerrs_1',u'fluxerrs_2',u'fluxerrs_3',u'fluxerrs_4',u'fluxerrs_5',u'fluxes_0',u'fluxes_1',
    #                       u'fluxes_2',u'fluxes_3',u'fluxes_4',u'fluxes_5',u'mjds_0',u'mjds_1',u'mjds_2',u'mjds_3',u'mjds_4',
    #                       u'mjds_5']].values

    zp_data = data.loc[:, [u'fluxerrs_0',
                           u'fluxerrs_1', u'fluxerrs_2', u'fluxerrs_3', u'fluxerrs_4', u'fluxerrs_5', u'fluxes_0',
                           u'fluxes_1',
                           u'fluxes_2', u'fluxes_3', u'fluxes_4', u'fluxes_5', u'mjds_0', u'mjds_1', u'mjds_2',
                           u'mjds_3', u'mjds_4',
                           u'mjds_5']].values
    '''
    zp_data = data.loc[:, [u'fluxes_0',u'fluxes_1', u'fluxes_2', u'fluxes_3', u'fluxes_4', u'fluxes_5']].values
    '''
    ##Zero-padding using Numpy and reshape in 1d vector [:,data]
    for item in zp_data:
        for a in item:
            if isinstance(a, float):
                #print(a)
                print(item,a)
    zp_data = np.asarray(
        [[np.pad(a*1./np.amax(a), (0, max_len - len(a)), 'constant', constant_values=0) for a in item] for item in zp_data])
    zp_data = zp_data.reshape(zp_data.shape[0], -1)
    zp_data = np.c_[
        zp_data, data.loc[:, [u'gal_b', u'gal_l', u'hostgal_photoz', u'hostgal_photoz_err', u'hostgal_specz']].values]
    ##Load labels and convert to integer
    labels = data.loc[:, [u'target']].values
    labels = labels.flatten()
    labels_name = np.array([6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99])
    [np.place(labels, labels == labels_name[i], [i]) for i in range(len(labels_name))]
    #print(zp_data.shape)

    return[zp_data,labels]

#dataset_handling(data)