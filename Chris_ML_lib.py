import keras.backend as K
from itertools import product
import numpy as np

""" Loss function definition """
##Personalised loss function , example taken from https://github.com/keras-team/keras/issues/2115
##Test needed comparing categorical cross entropy to weigthed cat cross entropy (without weights)

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

def compute_weights(N_of_Classes, penalty_factor=20):
    "Creating Weights: the critical and arbitrary factor is the penalty_factor"
    w_array = np.ones((len(N_of_Classes)+1, len(N_of_Classes)+1))

    for c_p, c_t in product(range(len(w_array)-1), range(len(w_array)-1)):
        w_array[c_t, c_p] = 1 + N_of_Classes[c_p]/np.sum(N_of_Classes) * penalty_factor        #Penalising 1 False classified as 0 -> has to be higher than 1
    return w_array