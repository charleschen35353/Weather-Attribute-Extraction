import tensorflow as tf
import numpy as np
import random
from os import path
from utils import one_hot_encode, rgb2hsv, rgb2gray
from env import *


def load_npz(path, input_shape, label_shape, batch_size = 16, onehot_encode = True):
    print("Loading data from {} ...".format(path))
    ds = np.load(path)
    x, y = ds['x'], ds['y']
    x[...,2] = x[...,2]/255
    train_data = x[:int(0.8*x.shape[0])]
    test_data = x[int(0.8*x.shape[0]):]
    
    train_labels = y[:int(0.8*y.shape[0])]

    if onehot_encode:
        train_labels = one_hot_encode(train_labels)
    
    test_labels = y[int(0.8*y.shape[0]):]   
    if onehot_encode:
        test_labels = one_hot_encode(test_labels)
    
    def generator(data, labels, batch_size = batch_size):
        ind = 0
        while True:
            yield data[ind:ind+batch_size], labels[ind:ind+batch_size]
            if ind >= data.shape[0]: ind = 0
        
    train_dataset = generator(train_data, train_labels, batch_size)
    test_dataset = generator(test_data, test_labels, batch_size)
    print("Dataset Loaded.")
    print("Train dataset input shape: {} label shape: {}".format(train_data.shape, train_labels.shape))
    print("Test dataset input shape: {} label shape: {}".format(test_data.shape, test_labels.shape))
    
    
    return [train_dataset, test_dataset, train_data.shape[0], test_data.shape[0]]
 
def generate_generator_mask(generator, path, batch_size = 8, img_height = IMG_HEIGHT, img_width = IMG_WIDTH):

        gen_img = generator.flow_from_directory(path,
                                              classes = ["images"],
                                              target_size = (img_height,img_width),
                                              batch_size = batch_size,
                                              shuffle=True, 
                                              seed=7)

        gen_mask = generator.flow_from_directory(path,
                                              classes = ["masks"],
                                              target_size = (img_height,img_width),
                                              batch_size = batch_size,
                                              color_mode = 'grayscale',
                                              shuffle=True, 
                                              seed=7)
        while True:
                imgs, _ = gen_img.next() #in 255
                if TRAIN_PREPROC == "hsv":
                    imgs = rgb2hsv(imgs)
                elif TRAIN_PREPROC == "hsv_norm":
                    imgs = rgb2hsv(imgs, normalization = True)
                elif TRAIN_PREPROC == "gray":
                    imgs = np.expand_dims(rgb2gray(imgs),axis = -1)
                masks, _ = gen_mask.next()
                masks = masks.squeeze()
  
                yield imgs, masks #Yield both images and their mutual label


class MaskDataloader:
    def __init__(self, data_path,  batch_size = 16, test_exists = True):
        
        
       	train_imgen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 20,\
										width_shift_range = 0.15, height_shift_range = 0.15,\
										horizontal_flip = True, rescale = 1/255.0)
        val_imgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)
  
        self.train_generator = generate_generator_mask(train_imgen,
                                               path = path.join(data_path,"train"),
                                               batch_size=batch_size)       

        self.val_generator = generate_generator_mask(val_imgen,
                                              path = path.join(data_path,"val"),
                                              batch_size=batch_size)  
        
        self.test_generator = None
        if test_exists:
            self.test_generator = generate_generator_mask(val_imgen,
                                              path = path.join(data_path,"test"),
                                              batch_size=batch_size)              
        else:
            self.test_generator = generate_generator_mask(val_imgen,
                                              path = path.join(data_path,"val"),
                                              batch_size=batch_size)            

        
    def load_image(self, mode = "train"):
        if mode == "val":
            return next(self.val_generator)
        elif mode == "test":
            return next(self.test_generator)
        elif mode == "train":
            return next(self.train_generator)
    
    def load_dl(self):
        return [self.train_generator, self.val_generator, self.test_generator]
