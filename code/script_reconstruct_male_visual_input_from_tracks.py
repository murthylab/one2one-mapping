
# plots trajectories of male and female fruit fly for one small period of courtship

import sys
sys.path.append('../classes')

import class_reconstruct

import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### MAIN SCRIPT


## load tracks
if True:
	data_folder = '../data/'

	X_tracks_female = np.load(data_folder + 'X_tracks_female.npy')
	X_tracks_male = np.load(data_folder + 'X_tracks_male.npy')
		# (x/y, head/body, frames)

	num_frames = X_tracks_female.shape[-1]

	R = class_reconstruct.ReconstructStimulusClass()

## reconstruct stimuli
if True:
	for iframe in range(500):
		print('frame {:d}'.format(iframe))
		png_save_filepath = '../data/pngs_reconstructed_stimuli/frame{:d}'.format(iframe)
		R.reconstruct_image_of_female(X_tracks_female[:,:,iframe], X_tracks_male[:,:,iframe], png_save_filepath)


## create gif of reconstructed stimuli
if True:
	png_folderpath = '../data/pngs_reconstructed_stimuli/'
	savemovie_filename = '../figs/movie_of_reconstructed_visual_input'
	
	R.make_gif_from_images(png_folderpath, num_frames, savemovie_filename)



