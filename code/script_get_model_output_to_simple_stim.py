
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

num_models = 10


## load classes
if True:
	S = class_stimuli.StimulusClass(stimulus_folder=stimulus_folder)
	M = class_model.ModelClass()

## plot model output behavior over time
if True:
	f = plt.figure(figsize=(12,12))

	ylims = [[-0.05,0.25], [-0.02,0.02], [-3.5,3.5], [0,0.2], [0,0.4], [0,0.55]]
		# for model outputs

	for itask, stimulus_task in enumerate(stimulus_tasks):
		S.set_stimulus_task(stimulus_task)

		# plot stimulus visual parameters
		ipanel = itask+1
		plt.subplot(7,3,ipanel)
		rotations, sizes, positions = S.get_rotations_sizes_positions()
		rotations = (rotations - np.mean(rotations)) / (np.std(rotations) + 1e-5)
		sizes = (sizes - np.mean(sizes)) / (np.std(sizes) + 1e-5) + 0.05
		positions = (positions - np.mean(positions)) / (np.std(positions) + 1e-5) + 0.1
		times = np.arange(320) * 1/30
		plt.plot(times, rotations, '--', color='xkcd:orange', label='rotation', alpha=0.5)
		plt.plot(times, sizes, '--', color='xkcd:black', label='size', alpha=0.5)
		plt.plot(times, positions, '--', color='xkcd:blue green', label='position', alpha=0.5)
		plt.legend()
		plt.ylabel('stim. param. value')
		plt.title(stimulus_task)
		ipanel += 3

		# compute outputs across models
		x_batch = S.get_x_batch()
		outputs_across_models = []
		for imodel in range(num_models):
			print('{:s}, computing model {:d} outputs...'.format(stimulus_task, imodel))
			M.load_model(filetag='KO_model{:d}'.format(imodel), save_folder=saved_models_folder + 'KO_models/')
			output = M.get_predicted_output(x_batch)
			outputs_across_models.append(output)
			# Note: Ignore 'skipping variable loading for optimizer SGD', it's because the keras models were
			#		saved without the optimizer state. The model can be compiled and trained.

		# plot outputs
		for ioutput in range(M.num_output_vars):
			plt.subplot(7,3,ipanel)
			for imodel in range(num_models):
				output = outputs_across_models[imodel]

				plt.plot(times[11:], output[ioutput], color='xkcd:light red', alpha=0.5)

			plt.ylabel(M.output_variable_names[ioutput])
			plt.ylim(ylims[ioutput])

			if ipanel <= 6:
				plt.title('KO network outputs (10 models)')
			if ipanel >= 19:
				plt.xlabel('time (s)')

			ipanel += 3

		f.tight_layout()
		f.savefig('../figs/KO_model_outputs.pdf')


