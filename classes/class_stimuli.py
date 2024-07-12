

# simple sequences to test output of model and model LCs
#   varying female position, size, and rotation

import numpy as np

import zipfile
import pickle

from tensorflow.keras.preprocessing import image

from PIL import Image
import scipy.ndimage as ndimage

import os



class StimulusClass:

	def __init__(self, stimulus_folder=None):
		if stimulus_folder is None:
			self.stimulus_folder = '../stimuli/simple_seqs/'
		else:
			self.stimulus_folder = stimulus_folder

		self.stimulus_input_tasks = ['varyposition', 'varysize', 'varyrotation']
			# tasks for simple sequences


	def set_stimulus_task(self, stimulus_task):
		# loads stimulus input for given task
		#
		# INPUT:
		# stimulus_input_task: (string): varyposition, varysize, varyrotation

		self.stimulus_task = stimulus_task

		P = pickle.load(open(self.stimulus_folder + stimulus_task + '_stimulus_params_P.pkl', 'rb'))
		self.sizes = P['female_sizes']
		self.rotations = P['female_rotations']
		self.positions = P['female_positions']

		self.stimulus_archive = zipfile.ZipFile(self.stimulus_folder + '{:s}.zip'.format(stimulus_task), 'r')


	def get_x_batch(self):
		# creates x_batch for simple sequences (varysize,varyposition,varyrotation)

		num_LCembed_vars = 23

		num_images = self.rotations.size
		inds = np.arange(num_images)
		imgs_recentered = self.get_recentered_imgs_from_inds(inds)

		img_seqs = self.set_images_to_img_seq(imgs_recentered)

		num_lag_frames = 10

		batch_size = len(img_seqs)
		image_inputs = np.zeros((num_lag_frames, batch_size, 64, 228, 1))

		for iseq in range(batch_size):
			for ilag in range(num_lag_frames):
				image_inputs[ilag,iseq,:,:,0] = img_seqs[iseq][num_lag_frames-ilag-1,:,13:241]
				
		x_batch = {}
		for ilag in range(num_lag_frames):
			x_batch['visionnet' + str(ilag) + '_image_input'] = image_inputs[ilag]

		x_batch['mask_input'] = np.ones((batch_size, num_LCembed_vars))

		return x_batch


	def get_recentered_imgs_from_inds(self, inds):
		# returns specified images based on inds

		imgs = np.zeros((inds.size,64,256))

		for iind in range(inds.size):
			ind = inds[iind]
			img_tag = 'image{:d}.png'.format(ind)
			# img = image.load_img(self.stimulus_archive.open(img_tag), color_mode='grayscale')
			# img = image.img_to_array(img)
			img = Image.open(self.stimulus_archive.open(img_tag)).convert('L')
			imgs[iind] = np.array(img).astype('float') - 255. # recenter, subtracting the white background

		return imgs


	def get_raw_imgs(self, imgs_recentered):

		return np.copy(imgs_recentered) + 255


	def get_rotations_sizes_positions(self):
		return self.rotations, self.sizes, self.positions


	def set_images_to_img_seq(self, imgs):
		# given imgs array, parcel out to img_seqs (to then get an x_batch)
		#
		# INPUT:
		#	imgs: (num_imgs, 64, 256) re-centered images in chronological order
		# OUTPUT:
		#   img_seqs: (num_imgs-10,) list, where img_seqs[iseq]: (num_lags, 64, 256)

		num_lag_frames = 10

		num_seqs = imgs.shape[0] - num_lag_frames - 1

		img_seqs = [np.zeros((num_lag_frames,64,256)) for x in range(num_seqs)]

		for iseq in range(num_seqs):
			for ilag in range(num_lag_frames):
				iframe = iseq + ilag
				img_seqs[iseq][ilag] = imgs[iframe]

		return img_seqs

		