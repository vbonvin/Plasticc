from DataProcessing import *
import os

pickle_path = '/scratch/vbonvin/Plasticc/test_samples'
scaler_path = './QuantileTransformer2.pkl'
model_path = './boostnwelossinvfilter.h5'

save_repertory = './outputtest'
pickle_list = os.listdir(picklepath)

for name in pickle_list:
    print(name)
    os.system("python predicttestset2.py " + name)
