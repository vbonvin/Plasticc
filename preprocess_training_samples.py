import pandas as pd
import dask.dataframe as dd
# from tqdm import tqdm
import _pickle as pickle
import shutil
import os, sys
import glob
import argparse


datakw = "training"

def writepickle(obj, filepath, verbose=True, protocol=-1):
	pkl_file = open(filepath, 'wb')
	pickle.dump(obj, pkl_file, protocol)
	pkl_file.close()
	if verbose: print("Wrote %s" % filepath)


def readpickle(filepath, verbose=True, py3=True):
	pkl_file = open(filepath, 'rb')
	if py3:
		obj = pickle.load(pkl_file, encoding='latin1')
	else:
		obj = pickle.load(pkl_file)
	pkl_file.close()
	if verbose: print("Read %s" % filepath)
	return obj


# import the data in dask
df = dd.read_csv("data/training_set.csv")
dfm = dd.read_csv("data/training_set_metadata.csv")


# read the ids
ids_df = df["object_id"]
ids_dfm = dfm["object_id"]
unique_ids = list(set(ids_dfm.compute().values))
bands = [0, 1, 2, 3, 4, 5]


series_list = []
last_index = 0

if not os.path.isdir('temps/%s' % datakw):
	os.makedirs('temps/%s' % datakw)


# in case of crash...
allpkls = glob.glob(os.path.join('temps', datakw, '*.pkl'))
if len(allpkls) > 0:
	lastpkl = max(allpkls, key=os.path.getctime)
	print("Restarting from last save...")
	series_list = readpickle(lastpkl, verbose=True)
	last_index = int(lastpkl.split(".pkl")[0].split("_")[-1])

for i, u_id in enumerate(unique_ids):
	if i < last_index:
		continue

	mask_df = (ids_df == u_id)
	df_samples = df[mask_df]

	mask_dfm = (ids_dfm == u_id)
	dfm_sample = dfm[mask_dfm]

	dfm_sample = dfm_sample.compute().iloc[0].to_dict()
	df_samples = df_samples.compute()

	for b in bands:
		bmask = (df_samples["passband"] == b)
		dfm_sample["mjds_%s" % b] = df_samples[bmask]["mjd"].values
		dfm_sample["fluxes_%s" % b] = df_samples[bmask]["flux"].values
		dfm_sample["fluxerrs_%s" % b] = df_samples[bmask]["flux_err"].values
		dfm_sample["detected_%s" % b] = df_samples[bmask]["detected"].values

	series_list.append(pd.Series(dfm_sample))

	# save the list
	if (i + 1) % 100 == 0:
		writepickle(series_list, os.path.join('temps', datakw, 'first_%s.pkl' % str(i + 1)))

samples = pd.concat(series_list, axis=1).T
# convert what needs to be integer as integer
samples[["object_id"]] = samples[["object_id"]].astype(int)

# pickle the results for later use
writepickle(samples, os.path.join("training_samples_astest.pkl"))
shutil.rmtree(os.path.join('temps', datakw))