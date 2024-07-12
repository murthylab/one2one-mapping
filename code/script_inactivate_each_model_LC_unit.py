
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


## load classes
if True:
	S = class_stimuli.StimulusClass(stimulus_folder=stimulus_folder)
	M = class_model.ModelClass()

	imodel = 0
	M.load_model(filetag='KO_model{:d}'.format(imodel), save_folder=saved_models_folder + 'KO_models/')

## plot model output behavior over time
if True:
	f = plt.figure(figsize=(12,20))

	ylims = [[-3.5,3.5], [-0.05, 0.25], [-0.05, 0.2]]
		# for model outputs

	for itask, stimulus_task in enumerate(stimulus_tasks):
		S.set_stimulus_task(stimulus_task)

		# plot stimulus visual parameters
		ipanel = itask+1
		plt.subplot(25,3,ipanel)
		rotations, sizes, positions = S.get_rotations_sizes_positions()
		rotations = (rotations - np.mean(rotations)) / (np.std(rotations) + 1e-5)
		sizes = (sizes - np.mean(sizes)) / (np.std(sizes) + 1e-5) + 0.05
		positions = (positions - np.mean(positions)) / (np.std(positions) + 1e-5) + 0.1
		times = np.arange(320) * 1/30
		plt.plot(times, rotations, '--', color='xkcd:orange', label='rotation', alpha=0.5)
		plt.plot(times, sizes, '--', color='xkcd:black', label='size', alpha=0.5)
		plt.plot(times, positions, '--', color='xkcd:blue green', label='position', alpha=0.5)
		plt.legend()
		plt.ylabel('stim.')
		plt.title(stimulus_task)
		ipanel += 3

		# get stimuli/inputs
		x_batch = S.get_x_batch()

		# comput output without inactivation
		outputs_noinact = M.get_predicted_output(x_batch)

		plt.subplot(25,3,ipanel)
		ioutput = 0
		if stimulus_task == 'varyposition':
			ioutput = 2  # angular velocity
		elif stimulus_task == 'varysize':
			ioutput = 0  # forward velocity
		elif stimulus_task == 'varyrotation':
			ioutput = 4  # prob pslow
		plt.plot(times[11:], outputs_noinact[ioutput], 'k', label='no inact')
		plt.ylabel(M.output_variable_names[ioutput])
		plt.xticks([])
		plt.legend()
		ipanel += 3

		# two types of inactivation:
		#	1) only inactivate the chosen LC  (necessary)
		#	2) inactivate all LCs except chosen LC  (sufficient)
		for iLC in range(M.num_model_LC_units):
			print('{:s}, {:s}'.format(stimulus_task, M.LC_types[iLC]))

			plt.subplot(25,3,ipanel)
			inds_inactivated_LCs = np.zeros((M.num_model_LC_units,)) == 1
			inds_inactivated_LCs[iLC] = True
			outputs_necessary = M.get_predicted_silencted_output(x_batch, KO_type='knockout-centered', inds_inactivated_LCs=inds_inactivated_LCs)

			inds_inactivated_LCs = np.zeros((M.num_model_LC_units,)) == 0
			inds_inactivated_LCs[iLC] = False
			outputs_sufficient = M.get_predicted_silencted_output(x_batch, KO_type='knockout-centered', inds_inactivated_LCs=inds_inactivated_LCs)

			ioutput = 0
			if stimulus_task == 'varyposition':
				ioutput = 2  # angular velocity
			elif stimulus_task == 'varysize':
				ioutput = 0  # forward velocity
			elif stimulus_task == 'varyrotation':
				ioutput = 4  # prob pslow

			plt.plot(times[11:], outputs_noinact[ioutput], 'k')
			plt.plot(times[11:], outputs_necessary[ioutput], 'r', label='inact this LC')
			plt.plot(times[11:], outputs_sufficient[ioutput], 'g', label='inact all but this LC')

			if ipanel <= 9:
				plt.legend()

			if ipanel >= 72:
				plt.xlabel('time (s)')
			else:
				plt.xticks([])
				plt.yticks([])

			if stimulus_task == 'varyposition':
				plt.yticks(ticks=[0], labels=[M.LC_types[iLC]], ha='right', rotation=0)

			plt.ylim(ylims[itask])

			ipanel += 3

		f.savefig('../figs/inactivated_KO_model_outputs.pdf')


