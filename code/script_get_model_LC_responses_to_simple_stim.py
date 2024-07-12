
import sys
sys.path.append('../classes')

import class_model
import class_stimuli

import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### MAIN SCRIPT

stimulus_tasks = ['varyposition', 'varysize', 'varyrotation']

stimulus_folder = '../stimuli/simple_seqs/'
saved_models_folder = '../saved_models/'

LC_types = ['LC4', 'LC6', 'LC9', 'LC10a', 'LC10ad', 'LC10bc', 'LC10d', 'LC11', 'LC12', 'LC13', 'LC15', 'LC16', 'LC17', 'LC18', 'LC20', 'LC21', 'LC22', 'LC24', 'LC25', 'LC26', 'LC31', 'LPLC1', 'LPLC2'] 
  
LC_colors = np.array([[166,206,227], [31,120,180], [178,223,138], [227,26,28], [251,154,153], [51,160,44],
	[253,191,111], [255,127,0], [202,178,214], [106,61,154], [215,233,39], [177,89,40], [190,186,218], [188,128,189],
	[247,236,50],[141,211,199], [251,128,114], [128,177,211], [253,180,98], [103,190,102], [252,205,229], [110,110,110],
	[50,80,50], [0,0,0]]) / 255.

## load classes
if True:
	S = class_stimuli.StimulusClass(stimulus_folder=stimulus_folder)
	M = class_model.ModelClass()
	imodel = 0
	M.load_model(filetag='KO_model{:d}'.format(imodel), save_folder=saved_models_folder + 'KO_models/')

## plot model output behavior over time
if True:
	f = plt.figure(figsize=(12,12))

	for itask, stimulus_task in enumerate(stimulus_tasks):
		print('computing for {:s}...'.format(stimulus_task))
		
		S.set_stimulus_task(stimulus_task)

		plt.subplot(1,3,itask+1)
		irow = 0

		# plot stimulus visual parameters
		ipanel = itask+1
		rotations, sizes, positions = S.get_rotations_sizes_positions()
		rotations = rotations - np.mean(rotations)
		rotations = rotations / (np.max(np.abs(rotations))+1e-5) / 2
		sizes = sizes - np.mean(sizes)
		sizes = sizes / (np.max(np.abs(sizes))+1e-5) / 2
		positions = positions - np.mean(positions)
		positions = positions / (np.max(np.abs(positions))+1e-5) / 2

		times = np.arange(320) * 1/30
		plt.plot(times, rotations + irow, '--', color='xkcd:orange', label='rotation', alpha=0.5)
		irow = irow - 1
		plt.plot(times, sizes + irow, '--', color='xkcd:black', label='size', alpha=0.5)
		irow = irow - 1
		plt.plot(times, positions + irow, '--', color='xkcd:blue green', label='position', alpha=0.5)
		irow = irow - 2

		# compute model LC responses
		x_batch = S.get_x_batch()

		model_LC_responses = M.get_model_LC_responses(x_batch)

		alpha = 2.  # scale to normalize all responses (so we keep magnitudes of responses)
		for iLC in range(len(LC_types)):
			responses = model_LC_responses[iLC,:] - np.mean(model_LC_responses[iLC,:])
			responses = responses / alpha  # scales across all LCs
			plt.plot(times[11:], responses + irow, color=LC_colors[iLC])
			irow = irow - 1

		plt.ylim([-27, 1])
		plt.title(stimulus_task)

		ticks = [0,-1,-2] + [i for i in range(-3-23,-3)][::-1]

		labels=['rotation', 'size', 'position'] + LC_types

		plt.yticks(ticks=ticks, labels=labels, rotation=0, ha='right')
		plt.xlabel('time (s)')

		f.tight_layout()
		f.savefig('../figs/KO_model_LC_responses.pdf')


