"""
Clean the metadata from the extra line at the top...
"""

import os, sys, glob

datapaths = list(sorted(glob.glob("data/chunks/test_set_chunk*_metadata.csv")))


for dp in datapaths[1:]:
	with open(dp, 'r') as f:
		newpath = "data/chunks/test_set_metadata_"+dp.split('_metadata.csv')[0].split('test_set_')[1] + '.csv'
		with open(newpath, 'w') as nf:
			f.next() # skip header line
			for line in f:
				nf.write(line)


