"""
A bunch of useful functions
"""


import os, sys
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import random

def writepickle(obj, filepath, verbose=True, protocol=-1):
	"""
	I write your python object obj into a pickle file at filepath.
	If filepath ends with .gz, I'll use gzip to compress the pickle.
	Leave protocol = -1 : I'll use the latest binary protocol of pickle.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath, 'wb')
	else:
		pkl_file = open(filepath, 'wb')

	pickle.dump(obj, pkl_file, protocol)
	pkl_file.close()
	if verbose: print "Wrote %s" % filepath


def readpickle(filepath, verbose=True):
	"""
	I read a pickle file and return whatever object it contains.
	If the filepath ends with .gz, I'll unzip the pickle file.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath,'rb')
	else:
		pkl_file = open(filepath, 'rb')
	obj = pickle.load(pkl_file)
	pkl_file.close()
	if verbose: print "Read %s" % filepath
	return obj


def display_sample(series):
	"""
	Nice plot

	:param series:
	:return:
	"""

	band_indexes = np.arange(6)
	plt.figure(figsize=(10, 4 * len(band_indexes)))

	colors = ["royalblue", "seagreen", "crimson", "purple", "indianred", "gold"]

	for b in band_indexes:
		plt.subplot(len(band_indexes), 1, b + 1)
		plt.errorbar(x=series["mjds_%i" % b], y=series["fluxes_%i" % b], yerr=series["fluxerrs_%i" % b], fmt='o',
		             c=colors[b])

	plt.show()


def bootstrap_sample(sample, days_offset=5000):
	"""
	Give me a pandas series, I return a bootstrapped version of it

	:param sample: a series containing the info for a single sample, see preprocess_training_samples for its structure
	:param days_offset: int, uniform range of the random offset applied to the mjds
	:return: bootstrapped sample
	"""

	newsample = sample.copy(deep=True)
	day_shift = random.uniform(-days_offset, days_offset)

	# loop over the 6 filters
	for band_index in range(6):
		newsample["mjds_%i" % band_index] = newsample["mjds_%i" % band_index] + day_shift
		fluxerrs = newsample["fluxerrs_%i" % band_index]
		fshifts = np.array([random.gauss(0.0, f) for f in fluxerrs])
		newsample["fluxes_%i" % band_index] = newsample["fluxes_%i" % band_index] + fshifts

	return newsample


def epurate_sample(sample, ep_percent=10):
	"""
	Give me a pandas series, I return an epurated version of its light curves (I shoot some points)

	:param sample: a series containing the info for a single sample, see preprocess_training_samples for its structure
	:param ep_percent: float, percentage of the points to shoot
	:return: epurated sample
	"""

	newsample = sample.copy(deep=True)

	# loop over the 6 filters
	for band_index in range(6):
		nobs = len(newsample["mjds_%i" % band_index])

		# make sure we have at least 5 points we don't shoot
		nshoot = min(max(0, nobs-5), int(round(nobs / 100. * ep_percent)))
		inds = np.arange(nobs)
		random.shuffle(inds)
		inds_toshoot = inds[:nshoot]

		tokeep = [False if i in inds_toshoot else True for i in np.arange(nobs)]

		newsample["mjds_%i" % band_index] = newsample["mjds_%i" % band_index][tokeep]
		newsample["fluxes_%i" % band_index] = newsample["fluxes_%i" % band_index][tokeep]
		newsample["fluxerrs_%i" % band_index] = newsample["fluxerrs_%i" % band_index][tokeep]

	return newsample