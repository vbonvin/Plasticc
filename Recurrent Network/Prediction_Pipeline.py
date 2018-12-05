from util import *
import os

from ANN_bruteforce.DataProcessing import *
from Chris_ML_lib import compute_weights, w_categorical_crossentropy
from sklearn.externals import joblib
from keras.models import model_from_json

'''
picklepath=sys.argv[1]
scalerpath=sys.argv[2]
modelpath=sys.argv[3]
identificationpacket=sys.argv[4]
'''
pickle_path = '/scratch/vbonvin/Plasticc/test_samples'
scaler_path = './scaler.pkl'
model_path = './RNN_model.h5'
model_weights_path = './RNN_model_weights.h5'

save_repertory = './outputtest'
pickle_list = os.listdir(pickle_path)

""" Model """

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_weights.h5")
print("Loaded model from disk")


""" Loss function definition """

## Creating Weights: the critical and arbitrary factor is the penalty_factor
N_of_Classes = np.array([151., 495., 924., 1193., 183., 30., 484., 102., 981., 208., 370., 2313., 239., 175.])
penalty_factor = 20

w_array = compute_weights(N_of_Classes, penalty_factor=20)

loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)

for name in pickle_list:

    """ Data Preparation """
    init_data = util.readpickle(pickle_path + '/' + name)

    test_data = init_data.loc[:].copy(deep=True)

    [x_test, idn] = dataset_zeropadding_3D(test_data, training=False)

    scaler = joblib.load(scaler_path)
    for ii in range(len(scaler)):
        x_test[:, ii, :] = scaler[ii].transform(x_test[:, ii, :])


    loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)
    model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])

    prediction = model.predict(x_test)

    idn = np.reshape(idn, (-1, 1))

    output = np.concatenate((np.array(idn), prediction), axis=1)

    np.savetxt(save_repertory + '/' + name[12:-4] + '.csv', output, delimiter=",")
    np.save(save_repertory + '/' + name[12:-4], output)  # 12fortest
    print ('csv')
