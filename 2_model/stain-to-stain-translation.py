# Stain-to-Stain translation
# from numpy import load
from numpy import zeros
from numpy import ones
import numpy as np
# from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Input
import matplotlib.pyplot as plt
import datetime

import define_discriminator as dis
import define_generator as gen
import data_loader as dl

# the num_step variable is a zero array for saving the loss of the generator and discriminator 
# num_step is calculated (number of epochs * number of training images)
num_step = 90000

Disc_loss_real = np.zeros((num_step,), dtype='float32')
Disc_loss_fake = np.zeros((num_step,), dtype='float32')
Gen_loss = np.zeros((num_step,), dtype='float32')


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# generate samples and save as a plot and save the model
def summarize_performance(epoch, batch, g_model, n_samples=3):
	# select a sample of input images
	[X_realB, X_realA] =  dl.load_batch_test(n_samples, is_testing=True)
	# generate a batch of fake samples
	X_fakeB = g_model.predict(X_realA)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	path_image = '/results/'
	for i in range(n_samples):
		plt.imsave(path_image +'Input/%dI_%d_%d.tiff' % (epoch+1, batch+1, i+1), X_realA[i])
		plt.imsave(path_image +'Generated/%dG_%d_%d.tiff' % (epoch+1, batch+1, i+1), X_fakeB[i])
		plt.imsave(path_image +'Original/%dO_%d_%d.tiff' % (epoch+1, batch+1, i+1), X_realB[i])
	# save the generator model
	path_model = '/models/'
	filename1 = 'plot_%d_%d' % (epoch+1, batch+1)
	filename2 = 'model_%d_%d.h5' % (epoch+1, batch+1)
	g_model.save(path_model + filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
# train pix2pix models
def train(d_model, g_model, gan_model, n_epochs=15, n_batch=1):
	start_time = datetime.datetime.now()
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# calculate the number of training iterations
	# manually enumerate epochs
	y_real = ones((n_batch, n_patch, n_patch, 1))
	for i in range(n_epochs):
		# select a batch of real samples
		for batch_i, (X_realB, X_realA) in enumerate(dl.load_batch_train(n_batch)):
			# generate a batch of fake samples
			X_fakeB = g_model.predict(X_realA)
			y_fake = zeros((len(X_fakeB), n_patch, n_patch, 1))
			# update discriminator for real samples
			d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
			# update discriminator for generated samples
			d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
			# update the generator
			g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
			elapsed_time = datetime.datetime.now() - start_time
			# summarize performance
			print('>step: %d >epoch %d-%d >batch %d-%d, D_loss_real[%.3f]  D_loss_fake[%.3f]  G_loss[%.3f]  time: %s'
				% ((batch_i+1 + (i * bat_per_epo)) ,i+1, n_epochs, batch_i+1, bat_per_epo, d_loss1, d_loss2, g_loss, elapsed_time))
			#Save the loss values in the array
			Disc_loss_real[i+1] = d_loss1
			Disc_loss_fake[i+1] = d_loss2
			Gen_loss[i+1] = g_loss
			# summarize model performance
            # set the number of times the model and images are saved
			if (batch_i+1) % 500 == 0:
				summarize_performance(i, batch_i, g_model)
                
                
# calculate the number of training image
bat_per_epo = 3000
# define input shape based on the loaded dataset
img_rows = 256
img_cols = 256
channels = 3
img_shape = (img_rows, img_cols, channels)
# define the models
d_model = dis.define_discriminator(img_shape)
g_model = gen.define_generator(img_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, img_shape)
# train model
train(d_model, g_model, gan_model)


