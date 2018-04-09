#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import networkx as nx
import random
import math
import pandas as pd
import scipy as sp
import copy
from scipy import stats
# could be omitted
from collections import deque
import time		
#========== PARAMETERS ========== 
n = 1000 # nodes
m = 2 #
p = 0.001 # probability to choose existing meme
nn = 100000 # timesteps
memory_size = 10
track_memes_after = 100 # only save memes after this step
max_memes_track = 1000 # number of memes to track after track_memes_after
#================================ 
print("******** PARAMS *********")
print('Network: BA: n:{0}, m:{1}, memory:{2}'.format(n, m, memory_size))
print('Introduction rate: {0}'.format(p))
print('Total time steps in this simulation: {0}'.format(nn))
print('Track memes after {0} steps'.format(track_memes_after))
print('Max memes to track: {0}'.format(max_memes_track))
print("*************************")

prefix = 'BA_n' + str(n) + '_m' + str(m) + '_mu' + str(p) + '_t' + str(nn) + '_memory'+str(memory_size)+'_'

G = nx.barabasi_albert_graph(n,m)
adj = G.adj # adjacency list

# network meme matrix: initialization
nodememory = sp.random.rand(n, memory_size)    #n*memory_size random [0,1) matrix. 
memes = {}
for meme in np.unique(nodememory):
	#temp = len( np.unique( np.where( np.any(nodememory == meme,1) )[0] ) )  #any effect?
	memes[meme] = {'start':1, 'end':1, \
				   'number_selected':0}
d = 10
progress = deque(list(np.linspace(1, d, d)*nn/d) + [nn-1])
starttime = time.time()

meme_count_after_saturation = 0
max_meme_size = 0
meme_dead=0
# begin simulation
print("Please wait while the simulation completes..")
t1 = time.time()
for counter in range(nn+1):

	counter = counter + 1
	if(counter >= progress[0]):
		print("{0}% done. Elapsed time: {1}".format(int(100 * progress[0]/nn), time.time()-t1))
		progress.popleft()
	select_one_node = random.randint(0,n-1)
	probability_new_idea = random.uniform(0,1) # mu
	if probability_new_idea <= p:
#################### Fitness ###################
		meme_chosen = random.random() #fitness for the new meme
		affectednodes = [select_one_node] + list(adj[select_one_node].keys())
#################### Fitness ###################
	else:
#################### select a meme from the list ###################
		meme_probability = random.uniform(0,1)
		temp = np.cumsum(nodememory[select_one_node, ])/sum(nodememory[select_one_node, ])
		meme_idx = np.where(temp >= meme_probability)[0][0]
		meme_chosen = nodememory[select_one_node, meme_idx] # selected meme
		affectednodes = list(adj[select_one_node].keys())
	
	temp = nodememory[affectednodes, :memory_size-1].copy()
	lastmemes = set(nodememory[affectednodes, memory_size-1])
	nodememory[affectednodes, 0] = meme_chosen
	nodememory[affectednodes, 1:] = temp
	# remove memes not found anywhere in the n/w
	for lm in lastmemes: 
		if (not np.any(nodememory == lm)) and (lm != meme_chosen) and (lm in memes):
			memes[lm]['end'] = counter
			if int(memes[lm]['start'])>=int(track_memes_after):
				meme_dead+=1
			if memes[lm]['start'] < track_memes_after:
				del memes[lm]
	if len(memes) > max_meme_size:
		max_meme_size = len(memes)
	# update meme set
	#max_pop = len(set(np.where(np.any(nodememory == meme_chosen, 1))[0]))
	if meme_chosen in memes:
		memes[meme_chosen]['lastnodeaffected'] = select_one_node
		memes[meme_chosen]['number_selected'] += 1
	else:
		if meme_count_after_saturation < max_memes_track:
			if counter >= track_memes_after:
				meme_count_after_saturation += 1
			memes[meme_chosen] = {'start':counter, 'end':1, \
								'number_selected':1}
		if meme_count_after_saturation + 1 == max_memes_track:
			print("Maximum no. of requested memes achieved at timestep: ", counter)
	
endtime = time.time()
print("Done! Total time in seconds: ", (endtime - starttime))	
# ==============================================================
# meme statistics
# ==============================================================
print("\nLen of memes set before final update: ", len(memes))
for m in list(memes.keys()):
	if memes[m]['start'] < track_memes_after:
		del memes[m]
print(len(memes), "out of", max_memes_track, "requested memes were born after step", track_memes_after)
print("Size of all memes (in MBs):", sys.getsizeof(memes)/float(1024**2))
print("Saving meme statistics..")
fitness = []
number_selected = []
for meme, val in memes.items():
	fitness.append(meme)
	number_selected.append(val['number_selected'])
memedata = pd.DataFrame(np.asarray([fitness, number_selected]).transpose(), columns=['Fitness', 'number_selected'])
memedata.to_csv(prefix + "meme.csv", index=False)
print("Max meme size:", max_meme_size)
print("Meme stats saved!")
