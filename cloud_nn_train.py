import glob
import argparse
from os import path
import matplotlib.image as pltimg
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import CloudSegWindow
from dataloaders import load_npz
from utils import rgb2hsv
from env import *

dataset_location = path.expanduser("~/dataset/sky_pixel_swim_seg_{}.npz".format(SLIDING_WINDOW_SIZE))

if CLOUD_FORM_DATA:
    img_dir = path.expanduser("~/dataset/swimseg/train/images") + "/*.png"
    mask_dir = path.expanduser("~/dataset/swimseg/train/masks") + "/*.png"
    img_paths = sorted(glob.glob(img_dir))
    mask_paths = sorted(glob.glob(mask_dir))
    data, labels = None, None
    n = 0
    for img_name, mask_name in zip(img_paths, mask_paths):
         print("Forming data set {} out of 920".format(n))
         img = pltimg.imread(img_name)
         img = cv2.resize(img, dsize = (256,256))
         mask = pltimg.imread(mask_name)
         mask = cv2.resize(mask, dsize = (256,256))
         img = rgb2hsv(img, normalization = True)
         patch_data, mask_data = [], []
         i = 0
         while i < img.shape[0]:
             j = 0
             while j < img.shape[1]:
                 patch = img[i:i+SLIDING_WINDOW_SIZE,j:j+SLIDING_WINDOW_SIZE,:]
                 padded = np.zeros(shape = ( SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE,3 ) )
                 padded[:patch.shape[0],:patch.shape[1]] = patch
                 patch_data.append(padded)
                 if MEAN_DATA:
                     mask_data.append([np.mean(mask[i:i+SLIDING_WINDOW_SIZE,j:j+SLIDING_WINDOW_SIZE]) > 0.5])
                 else:
                     patch = mask[i:i+SLIDING_WINDOW_SIZE,j:j+SLIDING_WINDOW_SIZE]
                     padded = np.zeros(shape = (SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE) )
                     padded[:patch.shape[0],:patch.shape[1]] = patch
                     mask_data.append(padded)
                 j+= int(SLIDING_WINDOW_SIZE/2)
             i+= int(SLIDING_WINDOW_SIZE/2)

         #sliding window classification
         patch_data = np.array(patch_data)
         mask_data = np.array(mask_data)
         if data is None:
             data = patch_data
         else:
             data = np.vstack((data,patch_data))

         if labels is None:
             labels = mask_data
         else:
             labels = np.vstack((labels,mask_data))
         n+=1

    data = data.astype(np.float32)
    labels = np.squeeze(labels).astype(np.int32)
    print(data.shape, end = ", ")
    print(data.dtype)
    print(labels.shape, end = ", ")
    print(labels.dtype)
    np.savez(dataset_location, x = data, y = labels)
    

batch_size = 4096
shuffle_buffer_size = 100
epochs = 200
train_dataset, test_dataset, train_size, test_size = load_npz(dataset_location, (SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE,3), (), batch_size = batch_size, onehot_encode = False)
checkpoint_path = "./cnn_ckpt/cloud_seg/cp.ckpt"
checkpoint_dir = path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model = CloudSegWindow(input_shape = (SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE,3))
model.load_weights(checkpoint_path)
keras.utils.plot_model(model, 'CloudNet.png')
print(model.summary())
model.fit_generator(train_dataset,
                    steps_per_epoch = train_size/batch_size,
                    epochs = epochs,
                    validation_data = test_dataset,
                    validation_steps = test_size/batch_size,
                    use_multiprocessing = False,
                    verbose = 1,
                    callbacks=[cp_callback])


