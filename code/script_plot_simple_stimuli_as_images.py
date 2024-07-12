
import sys
sys.path.append('../classes')

import class_stimuli

import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### MAIN SCRIPT

stimulus_tasks = ['varyposition', 'varysize', 'varyrotation']

stimulus_folder = '../stimuli/simple_seqs/'

## load classes
if True:
	S = class_stimuli.StimulusClass(stimulus_folder=stimulus_folder)


## plot stimuli as images
if True:
	f = plt.figure(figsize=(9,10))
	for itask, stimulus_task in enumerate(stimulus_tasks):
		S.set_stimulus_task(stimulus_task)

		imgs_recentered = S.get_recentered_imgs_from_inds(inds=np.arange(300,320))  # the last 20 frames
		imgs_raw = S.get_raw_imgs(imgs_recentered).astype('uint8')  # need to convert back to raw for visualization
		num_frames = imgs_raw.shape[0]
		ipanel = itask + 1
		for iframe in range(num_frames):
			plt.subplot(num_frames,3,ipanel)
			plt.imshow(imgs_raw[iframe], cmap='Greys_r')
			plt.xticks([]); plt.yticks([]);
			ipanel += 3  # skips to next row

			if iframe == 0:
				plt.title(stimulus_task)

	f.savefig('../figs/stimuli_as_images.pdf')




