from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from os import path
from tensorflow import keras
from models import SkySeg
from dataloaders import SkyDataloader
from env import *

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(tf.__version__)

batch_size = 64
trainset_size = 20210
valset_size = 2000
epochs = 500
data_path = "/home/charles/dataset/ADE20K/"
dl = SkyDataloader(data_path, test_exists = True)
train_generator, val_generator, _ = dl.load_dl()


checkpoint_path = "./sky_new/cp.ckpt"
checkpoint_dir = path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = SkySeg(input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHN), regularization_factor = RF, learning_rate = 7e-4)
print(model.summary())
#model.load_weights("./weights/cp.ckpt")
keras.utils.plot_model(model, 'Net.png')
	
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                  patience=20, min_lr=1e-5)
model.fit_generator(train_generator,
                                steps_per_epoch=trainset_size/batch_size,
                                epochs = epochs,
                                validation_data = val_generator,
                                validation_steps = valset_size/batch_size,
                                use_multiprocessing = False,
                                shuffle=True,
                                verbose = 1,
                                callbacks=[cp_callback])#, reduce_lr])

