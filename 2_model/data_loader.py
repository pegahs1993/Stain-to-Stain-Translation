import scipy.misc
from glob import glob
import numpy as np

img_rows = 256
img_cols = 256
channels = 3
img_shape = (img_rows, img_cols, channels)

dataset_name ='H_HG'
img_res=(img_rows, img_cols)

def load_batch_test(batch_size=1, is_testing=False):
	data_type = "train" if not is_testing else "test"
	path = glob('/MITOS/%s/%s/*' % (dataset_name, data_type))
	batch_images = np.random.choice(path, size=batch_size)
	imgs_A = []
	imgs_B = []
	for img_path in batch_images:
		img = imread(img_path)
		h, w, _ = img.shape
		_w = int(w/2)
		img_A, img_B = img[:, :_w, :], img[:, _w:, :]
		img_A = scipy.misc.imresize(img_A, img_res)
		img_B = scipy.misc.imresize(img_B, img_res)
		# If training => do random flip
		if not is_testing and np.random.random() < 0.5:
			img_A = np.fliplr(img_A)
			img_B = np.fliplr(img_B)
		imgs_A.append(img_A)
		imgs_B.append(img_B)
	imgs_A = np.array(imgs_A)/127.5 - 1.
	imgs_B = np.array(imgs_B)/127.5 - 1.
	return imgs_A, imgs_B

def load_batch_train( batch_size=1, is_testing=False):
	data_type = "train" if not is_testing else "val"
	path = glob('/MITOS/%s/%s/*' % (dataset_name, data_type))
	n_batches = int(len(path) / batch_size)
	for i in range(n_batches):
		batch = path[i*batch_size:(i+1)*batch_size]
		imgs_A, imgs_B = [], []
		for img in batch:
			img = imread(img)
			h, w, _ = img.shape
			half_w = int(w/2)
			img_A = img[:, :half_w, :]
			img_B = img[:, half_w:, :]
			img_A = scipy.misc.imresize(img_A, img_res)
			img_B = scipy.misc.imresize(img_B, img_res)
			if not is_testing and np.random.random() > 0.5:
				img_A = np.fliplr(img_A)
				img_B = np.fliplr(img_B)
			imgs_A.append(img_A)
			imgs_B.append(img_B)
		imgs_A = np.array(imgs_A)/127.5 - 1.
		imgs_B = np.array(imgs_B)/127.5 - 1.
		yield imgs_A, imgs_B
        
def imread(path):
	return scipy.misc.imread(path, mode='RGB').astype(np.float)