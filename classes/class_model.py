
import numpy as np

import os, sys
gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device

import tensorflow as tf
from tensorflow.keras import backend as K
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.utils.disable_interactive_logging()

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Flatten, Multiply, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D, DepthwiseConv2D

from tensorflow.keras.optimizers import SGD
from numpy.random import seed

from tensorflow.keras.models import load_model

import pickle, copy


class ModelClass: 
	# document
	#  10 models per model type

	
	def __init__(self, save_folder='../saved_models/'):
		# intializes class, determining what type of model
		#
		# INPUT:
		#	save_folder: (string), where model is saved

		self.save_folder = save_folder

		self.num_model_LC_units = 23
		self.output_variable_names = ['forward_vel', 'lateral_vel', 'angular_vel', 'prob_pfast', 'prob_pslow', 'prob_sine']
		self.num_output_vars = 6

		self.LC_types = ['LC4', 'LC6', 'LC9', 'LC10a', 'LC10ad', 'LC10bc', 'LC10d', 'LC11', 'LC12', 'LC13', 'LC15', 'LC16', 'LC17', 'LC18', 'LC20', 'LC21', 'LC22', 'LC24', 'LC25', 'LC26', 'LC31', 'LPLC1', 'LPLC2'] 


	def make_vision_network(self):
		# document

		base_name = 'visionnet_'

		num_layers_visionnet = 3
		num_filters_visionnet = 32
		num_embed_vars_visionnet = 16
		kernel_shape = (7,27)

		x_input = Input(shape=(64,228,1), name=base_name + 'image_input')
		x = x_input

		ilayer = 0
		x = Conv2D(filters=num_filters_visionnet, kernel_size=3, strides=2, padding='same', name=base_name + 'layer{:d}conv'.format(ilayer))(x)
		x = BatchNormalization(axis=-1, name=base_name + 'layer{:d}_batch'.format(ilayer))(x)
		x = Activation(activation='relu', name=base_name + 'layer{:d}_act'.format(ilayer))(x)

		for ilayer in range(num_layers_visionnet-1):
			x = SeparableConv2D(filters=num_filters_visionnet, kernel_size=3, strides=2, padding='same', name=base_name + 'layer{:d}conv'.format(ilayer+1))(x)
			x = BatchNormalization(axis=-1, name=base_name + 'layer{:d}_batch'.format(ilayer+1))(x)
			x = Activation(activation='relu', name=base_name + 'layer{:d}_act'.format(ilayer+1))(x)

		x = DepthwiseConv2D(kernel_size=(8,29), strides=1, padding='valid', name=base_name + 'spatial_pool_layer')(x)
		x = Flatten()(x)
		x = Dense(units=num_embed_vars_visionnet, name=base_name + '_embed_vars')(x)
		
		return Model(x_input, x)


	def initialize_model(self, learning_rate=1e-2):	
		# DOCUMENT

		num_layers_decisionnet = 3
		num_filters_decisionnet = 128

		# model hyperparameters
		self.learning_rate = learning_rate
		self.momentum = 0.7
		self.decay = 0.
		self.num_lag_frames = 10

		self.fly_types = ['LC4', 'LC6', 'LC9', 'LC10a', 'LC10ad', 'LC10bc', 'LC10d', 'LC11', 'LC12', 'LC13', 'LC15', 'LC16', 'LC17', 'LC18', 'LC20', 'LC21', 'LC22', 'LC24', 'LC25', 'LC26', 'LC31', 'LPLC1', 'LPLC2', 'PDB'] 

		x_inputs_vision_model_list = []

		x_input_mask = Input(shape=(self.num_model_LC_units,), name='mask_input')

		# get visionnet (shared across frames)
		if True:

			# make num_lag_frames vision models (for previous num_lag_frames frames)
			self.visionnet_model = self.make_vision_network()

			x_vision_model_list = []

			for ivision_model in range(self.num_lag_frames):
				x_input = Input(shape=(64,228,1), name='visionnet{:d}_image_input'.format(ivision_model))
				x = self.visionnet_model(x_input)
				x_inputs_vision_model_list.append(x_input)
				x_vision_model_list.append(x)

			x_input_visionnet_embed_vars_across_frames = [] + x_vision_model_list
			x_conc = Concatenate(axis=-1)(x_input_visionnet_embed_vars_across_frames)

		# prepare inputs for full network  (could include mask, etc. here)
		x_inputs = x_inputs_vision_model_list + [x_input_mask]

		# make embedding layer (integrating over visionnet's outputs across frames)
		if True:
			x = Dense(units=64, name='embedding_layer_dense1')(x_conc)
			x = BatchNormalization(axis=-1, name='embedding_layer_batchnorm1')(x)
			x = Activation(activation='relu', name='embedding_layer_act1')(x)

			x = Dense(units=self.num_model_LC_units, name='embedding_layer_dense2')(x)
			x = BatchNormalization(axis=-1, name='embedding_layer_batchnorm2')(x)
			x = Activation(activation='relu', name='embedding_layer')(x)

			x = Multiply()([x, x_input_mask])  # apply mask to LC neurons
		
		# prepare decision network
		if True:
			for ilayer in range(num_layers_decisionnet):
				x = Dense(units=num_filters_decisionnet, name='decision_dense{:d}'.format(ilayer))(x)
				x = BatchNormalization(axis=-1, name='decision_batch{:d}'.format(ilayer))(x)
				x = Activation(activation='relu', name='decision_act{:d}'.format(ilayer))(x)

		x_forward_vel = Dense(units=1, name='forward_vels')(x)
		x_lateral_vel = Dense(units=1, name='lateral_vels')(x)
		x_angular_vel = Dense(units=1, name='angular_vels')(x)
		x_pfast_pulse_bits = Dense(units=1, activation='sigmoid', name='pfast_pulse_bits')(x)
		x_pslow_pulse_bits = Dense(units=1, activation='sigmoid', name='pslow_pulse_bits')(x)
		x_sine_bits = Dense(units=1, activation='sigmoid', name='sine_bits')(x)

		x_outputs = []
		x_outputs.append(x_forward_vel)
		x_outputs.append(x_lateral_vel)
		x_outputs.append(x_angular_vel)
		x_outputs.append(x_pfast_pulse_bits)
		x_outputs.append(x_pslow_pulse_bits)
		x_outputs.append(x_sine_bits)

		self.model = Model(inputs=x_inputs, outputs=x_outputs)

		losses = {
				'forward_vels': 'mean_squared_error',
				'lateral_vels': 'mean_squared_error',
				'angular_vels': 'mean_squared_error',
				'pfast_pulse_bits': 'binary_crossentropy',
				'pslow_pulse_bits': 'binary_crossentropy',
				'sine_bits': 'binary_crossentropy'
				}

		optimizer = SGD(learning_rate=self.learning_rate, weight_decay=self.decay, momentum=self.momentum)

		self.model.compile(optimizer=optimizer, loss=losses)


	def get_predicted_output(self, x_batch, unnormalize_flag=True):
		# get predicted output behavior of model
		#
		# INPUT:
		#	x_batch: (dict from stimulus class), dict of predictants and images for one batch of stimuli (with num_batch_samples samples)
		#	unnormalize_flag: (True or False), converts output into raw movement space
		#			(networks trained on z-scored data)
		# OUTPUT:
		#	output: (list), where output[0] --> forward vel, output[1] --> lateral vel,
		#		output[2] --> angular vel, output[3] --> pfast prob, output[4] --> pslow prob
		#		output[5] --> sine song prob
		#		output[ivar]: (num_frames,1)

		output = self.model.predict(x_batch, verbose=False)

		if unnormalize_flag == True:
			output[0] = output[0] * 0.131218 + 0.056153  # now in mm/s
			output[1] = output[1] * 0.065737 + 0.001365  # now in mm/s
			output[2] = output[2] * 6.372305 + 0.052968  # now in vis deg/s

		return output


	def get_model_LC_responses(self, x_batch):
		# get LC embeddings for one batch
		# 
		# INPUT:
		#	x_batch: (dict from stimulus class), dict of predictants and images for one batch of stimuli (with num_batch_samples samples)
		# OUTPUT:
		#	responses: (num_model_LC_units, num_batch_samples)

		model_input_to_responses = Model(inputs=self.model.input, outputs=self.model.get_layer('embedding_layer').output)

		return model_input_to_responses.predict(x_batch, verbose=False).T


	def save_model(self, filetag='model', save_folder=None):
		# saves model to memory (as keras model)
		#
		# INPUT:
		#	filetag: (string), filename of model (without .keras), e.g., 'KO_model0'
		#	save_folder: (string), path of model (ends with /), e.g., '../saved_models/'
		#		if None, sets to self.save_folder
		# OUTPUT:
		#	None. 

		if save_folder is None:
			save_folder = self.save_folder

		self.model.save(save_folder + filetag + '.keras')


	def load_model(self, filetag, save_folder=None):
		# loads model from memory (as keras model)
		#
		# INPUT:
		#	filetag: (string), filename of model (without .keras), e.g., 'KO_model0'
		#	save_folder: (string), path of model (ends with /), e.g., '../saved_models/'
		#		if None, sets to self.save_folder
		#		example: filetag='KO_model0', save_folder='./saved_models/KO_models/'
		# OUTPUT:
		#	None. 

		if save_folder is None:
			save_folder = self.save_folder

		K.clear_session()

		self.model = load_model(save_folder + filetag + '.keras')


	def get_predicted_silencted_output(self, x_batch, KO_type='knockout', inds_inactivated_LCs=None, unnormalize_flag=True):
		# predicted output given a subset of model LC units are inactivated
		#	There are two ways to inactivate:
		#	'knockout' --> activity of model LC is set to 0
		#	'knockout-centered' --> activity of model LC is set to its constant mean
		#	The latter better helps to identify the model LC's contribution to behavior,
		#	as it keeps the activity in its working regime. We use the latter in Figure 4.
		#
		# INPUT:
		#	x_batch: (dict from stimulus class), dict of predictants and images for one batch of stimuli (with num_batch_samples samples)
		#	KO_type: ('knockout', 'knockout-centered'), type of knockout, see above
		#	inactivated_LCs: (binary True/False 1d array of length = num model LC units),
		#		e.g., [F,T,F,F,...,F] indicates model LC unit 1 to be inactivated
		#	unnormalize_flag: (True or False), converts output into raw movement space
		#			(networks trained on z-scored data)
		# OUTPUT:
		#	output: (list), output behavior, see 'get_predicted_output' for details

		if inds_inactivated_LCs is None:
			inds_inactivated_LCs = np.zeros((self.num_model_LC_units,)) == 1

		if KO_type == 'knockout':
			x_batch['mask_input'][:,inds_inactivated_LCs] = 0.
			output = self.get_predicted_output(x_batch, unnormalize_flag=False)
			x_batch['mask_input'] = np.ones(x_batch['mask_input'].shape)  # reset to all ones
		else:  # KO_type == 'knockout-centered'
			# we need to first get model LC responses to stimulus:
			model_LC_responses = self.get_model_LC_responses(x_batch)
				# (num_LC_types, num_frames)

			# inactivate chosen model LCs
			means = np.mean(model_LC_responses,axis=1)
			for iLC in range(self.num_model_LC_units):
				if inds_inactivated_LCs[iLC] == True:
					model_LC_responses[iLC,:] = means[iLC]

			# get output
			model_responses_to_output = self.get_model_LC_responses_to_output()
					# need to build a model of the decision network
			output = model_responses_to_output.predict(model_LC_responses.T)

		if unnormalize_flag == True: # set units back to original raw movement space
			output[0] = output[0] * 0.131218 + 0.056153  # now in mm/s
			output[1] = output[1] * 0.065737 + 0.001365  # now in mm/s
			output[2] = output[2] * 6.372305 + 0.052968  # now in vis deg/s

		return output


	def get_model_LC_responses_to_output(self):
		# retrieves a model that maps model LC responses to output (i.e., the decision network)
		#
		# INPUT:
		#	None. Uses class's internal variables
		# OUTPUT:
		#	model_responses_to_output: (keras model instance), model that takes model LC responses as input
		#			and outputs behavior as a list

		num_layers_decisionnet = 3
		num_filters_decisionnet = 128

		x_input = Input(shape=(self.num_model_LC_units,), name='z_input')
		x = x_input

		output_names = ['forward_vels', 'lateral_vels', 'angular_vels', 'pfast_pulse_bits', 'pslow_pulse_bits', 'sine_bits']
		
		# prepare decision network
		for ilayer in range(num_layers_decisionnet):
			x = Dense(units=num_filters_decisionnet, name='z_dense_layer{:d}'.format(ilayer))(x)
			x = BatchNormalization(axis=-1, name='z_bn_layer{:d}'.format(ilayer))(x)
			x = Activation(activation='relu', name='z_act_layer{:d}'.format(ilayer))(x)

		x_outputs = []
		for ioutput in range(len(output_names)):
			if ioutput < 3:
				x_out = Dense(units=1, name='z_{:s}'.format(output_names[ioutput]))(x)
			else:
				x_out = Dense(units=1, activation='sigmoid', name='z_{:s}'.format(output_names[ioutput]))(x)

			x_outputs.append(x_out)

		model_responses_to_output = Model(inputs=x_input, outputs=x_outputs)

		# set weights to trained model's weights
		for ilayer in range(num_layers_decisionnet):
			weights = self.model.get_layer('decision_dense{:d}'.format(ilayer)).get_weights()
			model_responses_to_output.get_layer('z_dense_layer{:d}'.format(ilayer)).set_weights(copy.deepcopy(weights))

			weights = self.model.get_layer('decision_batch{:d}'.format(ilayer)).get_weights()
			model_responses_to_output.get_layer('z_bn_layer{:d}'.format(ilayer)).set_weights(copy.deepcopy(weights))

		for ioutput in range(len(output_names)):
			weights = self.model.get_layer(output_names[ioutput]).get_weights()
			model_responses_to_output.get_layer('z_{:s}'.format(output_names[ioutput])).set_weights(copy.deepcopy(weights))

		return model_responses_to_output





