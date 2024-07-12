import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms

import numpy as np

from tensorflow.keras.preprocessing import image

from PIL import Image

import zipfile, os

# Reconstructs an image given the tracks of female and male fruit fly

class ReconstructStimulusClass: 

	def __init__(self):

		self.fig, self.ax = plt.subplots()

		self.imgs = []

		self.head_radius = 2/3  # hyperparameters defining size of fictive female
		self.tail_radius = 0.5
		self.ellipse_radius = 2.

		self.image_width = 4.  # for stripes model


	### HELPER FUNCTIONS

	# helper function
	def resize_image(self, imgsave_filename):
		# resize jpg
		im = Image.open(imgsave_filename + '.png')
		im = im.resize((256,64), Image.LANCZOS)
		im.save(imgsave_filename + '.png')


	# helper function
	def get_X_tracks_female_relative_to_male(self, X_tracks_female, X_tracks_male):
		# retrieves female joint positions relative to male's head
		# INPUT:
		#	X_tracks_female: (2,3) --> (x/y,head/body/tail), joint positions of female
		#	X_tracks_male: (2,3) --> (x/y,head/body/tail), joint positions of female
		# OUTPUT:
		#	X_tracks_female_relative: (2,3) --> (x/y, head/body/tail), joint positions of female
		#		relative to male's head (e.g., female positions minus male's head position)

		X_tracks_female_relative = X_tracks_female - X_tracks_male[:,0][:,np.newaxis]

		return X_tracks_female_relative


	# helper function
	def get_head_dir(self, X_tracks):
		# INPUT:
		#	X_tracks: (2,3) positions head/body/tail, x/y
		# OUTPUT:
		#	head_dir: (2,) estimated head dir for each frame

		ihead=0; ibody=1; itail=2;
		head_dir = X_tracks[:,ihead] - X_tracks[:,ibody]
		head_dir = head_dir / (np.sqrt(np.sum(head_dir**2)) + 1e-5)

		return head_dir


	# helper function
	def get_orth_head_dir(self, head_dir):
		# finds orth head dir (90 degrees clockwise)
		# INPUT:
		#   head_dir (2,)  original head dir (normalized)
		# OUTPUT:
		#	orth_head_dir (2,) 90 degree clockwise rotation (to the fly's right)

		orth_head_dir = np.zeros((2,)) # rotation matrix causes [x,y] to have orth vector: [y,-x], 90 degree clockwise rotation
		orth_head_dir[0] = head_dir[1]
		orth_head_dir[1] = -head_dir[0]  
		orth_head_dir = orth_head_dir / (np.sqrt(np.sum(orth_head_dir**2)) + 1e-5)

		return orth_head_dir


	# helper function
	def get_position_along_circumfrence(self, head_dir, orth_head_dir, X_tracks_female_relative_body):
		#	identifies female position in visual degrees (normalized between -1 and 1)
		# INPUT:
		#	head_dir: (2,), head direction of male fly
		#	X_tracks_female_relative_body: (2,), female position relative to male's head
		# OUTPUT:
		#	position_along_circum: (1,), lateral position of the female, where
		#		where -0.5 is to the left, 0.5 is to the right, {-1/1} is directly behind, 0 is in front of male

		# get angle between head_dir and pos_female
		pos = np.copy(X_tracks_female_relative_body) / (np.sqrt(np.sum(X_tracks_female_relative_body**2)) + 1e-6)

		angle_head_dir_pos_female = np.arccos(np.sum(head_dir * pos))

		# get proj onto orth head_dir
		proj_orth = np.sum(orth_head_dir * pos)

		if (proj_orth < 0):   # if female on left side, make angle negative
			angle_head_dir_pos_female = -angle_head_dir_pos_female

		# find position along circumfrence (between 0 and 1)
		position_along_circum = angle_head_dir_pos_female / np.pi
		# negative 1 is furthest left, positive 1 is furthest right (both are directly behind fly)

		return position_along_circum


	# helper function
	def get_radius(self, diff_pos_joint_female, joint='head'):
		# returns radius in terms of 1/pi
		
		if joint == 'head':
			joint_radius = self.head_radius
		else:
			joint_radius = self.tail_radius

		dist_between_flies = np.sqrt(np.sum((diff_pos_joint_female)**2)) + 1e-6

		# find radius of object, which is akin to finding angle between center and outer edge of circle
		radius = np.arctan(joint_radius / dist_between_flies) * 2.0

		return radius / np.pi


	# helper function
	def get_params(self, X_tracks_female_relative, head_dir_male, orth_head_dir_male, head_dir_female):
		# width is a fixed distance in xy space (self.image_width) that I am assuming is the female fly's width (~4 mm)
		#	height is simply a scalar of this
		# DOCUMENT

		ibody = 1
		diff_pos_body_female = X_tracks_female_relative[:,ibody]

		# lateral position (0 is front facing, negative is to the left, positive is to the right)
		lateral_pos = self.get_position_along_circumfrence(head_dir_male, orth_head_dir_male, diff_pos_body_female)


		# width (use arctan for this)
		dist = np.sqrt(np.sum(diff_pos_body_female**2)) + 1e-6
		view_width = np.arctan(self.image_width /2.0 / dist) / np.pi * 4.0  # * 2 b/c tan^-1 returns in range 0 to pi/2 and another 2 b/c we halved the width

		view_height = 0.46 * view_width # 0.46 scalar b/c image height (225 pixels) is 0.46 of image width (490 pixels)
												# not used for stripes model

		return (lateral_pos, view_height, view_width)


	# helper function
	def get_visual_degree_female_image(self, head_dir_male, orth_head_dir, head_dir_female, X_tracks_female_relative_body):
		# computes the angle (in degrees) for the femalefly360 image
		# INPUT:
		#	head_dir_male: (2,), male's body->head dir in absolute coordinates
		#	orth_head_dir: (2,), dir orthogonal to male head dir (90 degrees clockwise assuming we are looking down on top of male)
		#			e.g., if head_dir_male=[1,0], then orth_head_dir=[0,-1]
		#	head_dir_female: (2,), female's body->head dir in absolute coordinates
		#	X_tracks_female_relative_body: (2,), relative vector between male's head position and female's body position in absolute coordinates
		# OUTPUT:
		#	angle: (integer between -180 and 180), rotation angle for the female fly
		#		angle of 0 --> female faces away from male
		#		angle of -90 --> female faces to the left
		#		angle of 90 --> female faces to the right
		#		angle of -180/180 --> female faces towards male
		# idea: compute angle between vector of male-female positions and female head direction
		#   step 1: rotate to male-centric coordinates (male facing towards (1,0))
		#   step 2: identify direction orthogonal to the relative body direction
		#   step 3: compute angle between female head dir and relative body direction

		# project head_dir_female and X_tracks_female_relative_body onto head_dir_male coordinates
		# so that male is now facing (1,0)
		if True:
			x_new = np.copy(head_dir_female)
			x_new[0] = np.dot(head_dir_male.T, head_dir_female)
			x_new[1] = np.dot(orth_head_dir.T, head_dir_female)
			head_dir_female = x_new

			x_new = np.copy(X_tracks_female_relative_body)
			x_new[0] = np.dot(head_dir_male.T, X_tracks_female_relative_body)
			x_new[1] = np.dot(orth_head_dir.T, X_tracks_female_relative_body)
			X_tracks_female_relative_body = x_new

			rel_body_direction = X_tracks_female_relative_body / (np.sqrt(np.sum(X_tracks_female_relative_body**2)) + 1e-6)

		# compute angle between female head dir and relative body direction
		#   to see if female is facing towards/away from male
		p = np.dot(head_dir_female.T, rel_body_direction)
		p = np.clip(p, a_min=-1, a_max=1)

		# identify if female is facing to the left/right of male
		if True:
			# find orthogonal projection of rel_body_direction (90 degree clockwise)
			orth_rel_body_direction = -self.get_orth_head_dir(rel_body_direction)
			  # negate here b/c we rotated the coordinates so male faces 
			  #  the right (1,0) and the male's right is down but considered (0,1)..which is usually (0,-1)
			  #  so you need to negate the orth dir (b/c here anti-clockwise is positive)
			  #  (this is a very subtle point but important for the signage)

			# compute sign --> positive if female faces toward the right, negative if to the left
			p_orth = np.dot(head_dir_female.T, orth_rel_body_direction)
			p_orth = np.clip(p_orth, a_min=-1, a_max=1) # clips in case of numerical problems
			signer = np.sign(p_orth)
			if signer == 0:
			  signer = 1

		# compute final angle
		angle = signer * np.arccos(p) / np.pi * 180
		  # signer --> denotes if female faces to the left (negative) or right (positive)

		angle = np.round(angle).astype(int)
		angle = np.clip(angle, a_min=-180, a_max=180)

		return angle


	def reconstruct_image_of_female(self, X_tracks_female, X_tracks_male, png_save_filepath):
		# reconstructs image (with respect to male heading dir) given male and female positions
		#    - this is for one image. Saves as png (no loss).
		#	 Note: Uses the pngs of a fictive female with different orientations in './data/stripes360deg/'
		# INPUT:
		#	X_tracks_female: (2,3) for (x/y, head/body/tail) positions of female fly for one frame
		#	X_tracks_male: (2,3) for (x/y, head/body/tail) positions of male fly for one frame (same as female)
		# 	png_save_filepath: (string), location to save png file (one per frame) --- folder must exist!
		#			e.g., './data/pngs_reconstructed_stimuli/frame0'
		# OUTPUT:
		#   None.  (saves new image as png in save_filepath)

		if X_tracks_female.ndim > 2 or X_tracks_male.ndim > 2:
			raise ValueError('reconstruct_image_of_female() only accepts one frame at a time; check shapes of X_tracks_female and X_tracks_male')

		fig, ax = plt.subplots()
		fig.set_size_inches(10,2.5)
		ax.set_xlim([-1, 1])
		ax.set_ylim([-0.25, 0.25])

		ix=0; iy=1;
		ihead=0; ibody=1; itail=2;

		# collect head directions
		head_dir_male = self.get_head_dir(X_tracks_male)
		orth_head_dir = self.get_orth_head_dir(head_dir_male)

		X_tracks_female_relative = self.get_X_tracks_female_relative_to_male(X_tracks_female, X_tracks_male)
		head_dir_female = self.get_head_dir(X_tracks_female_relative)

		angle_3dmodel = self.get_visual_degree_female_image(head_dir_male, orth_head_dir, head_dir_female, X_tracks_female_relative[:,1])

		# load image
		img_filepath = '../data/stripes360deg/fly{:d}.png'.format(angle_3dmodel)
			# assumes you are calling from the ./code/ folder

		im = Image.open(img_filepath)
		img = np.array(im)

		# get position, height, and width
		(lateral_pos, view_height, view_width) = self.get_params( 
							X_tracks_female_relative, head_dir_male,
							orth_head_dir, head_dir_female)

		left = lateral_pos - view_width/2
		right = lateral_pos + view_width/2
		top = view_height/2
		bottom = -view_height/2

		ax.imshow(img, extent=(left,right,bottom,top))

		bbox = transforms.Bbox([[1.3,0.3],[8.95,2.15]])

		z = 1.
		ax.set_facecolor((z,z,z))

		fig.savefig(png_save_filepath + '.png', bbox_inches=bbox)

		plt.close('all')

		self.resize_image(png_save_filepath)


	def make_gif_from_images(self, png_folderpath, num_frames, savegif_filename):
		# makes a .gif of the sequence of images of the fictive female
		# INPUT:
		#	png_folderpath: (str), folderpath of where the pngs are stored, e.g., './data/pngs_reconstructed_stimuli/'
		#		each png should be saved as "frameXX.png", where XX varies from 0 to num_frames-1
		#	num_frames: (int), number of frames
		#	savegif_filename: (str), filename to store gif, e.g., 'reconstructed_visual_input_sequence'
		# OUTPUT:
		#	None. Saves a gif in desired savegif_filename.

		images = []
		for iframe in range(num_frames):
			png_filename = 'frame{:d}.png'.format(iframe)
			img = Image.open(png_folderpath + png_filename)
			images.append(img)

		duration = 1/30 # inter-frame interval in seconds
		images[0].save(savegif_filename + '.gif', save_all=True, append_images=images[1:], duration=duration,loop=0)

