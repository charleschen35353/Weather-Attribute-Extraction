import glob
import argparse
from os import path
import matplotlib.image as pltimg
import cv2
import numpy as np
from utils import rgb2hsv
from env import *

dataset_location = path.expanduser("~/dataset/sky_pixel_swim_test.npz")

if SVM_FORM_DATA:
    img_dir = path.expanduser("~/dataset/swimseg/train/images") + "/*.png"
    mask_dir = path.expanduser("~/dataset/swimseg/train/masks") + "/*.png"
    img_paths = sorted(glob.glob(img_dir))
    mask_paths = sorted(glob.glob(mask_dir))
    data = None
    labels = None
    for img_name, mask_name in zip(img_paths, mask_paths):
         print("Forming data set {} out of 200".format(n))
         img = pltimg.imread(img_name)
         img = cv2.resize(img, dsize = (256,256))
         mask = pltimg.imread(mask_name)
         mask = cv2.resize(mask, dsize = (256,256))
         img = rgb2hsv(img)
         img_int = (img[...,2]*255).astype(np.uint8)
         img_int = cv2.equalizeHist(img_int)
         img[...,2] = img_int
     
         #pixel classification

         patch_data, mask_data = [], []
         i = 0
         while i < img.shape[0]:
             j = 0
             while j < img.shape[1]:
                 patch = img[i:i+SLIDING_WINDOW_SIZE,j:j+SLIDING_WINDOW_SIZE,:]
                 padded = np.zeros(shape = ( SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE ,3 ) )
                 padded[:patch.shape[0],:patch.shape[1]] = patch
                 padded = np.reshape(padded, (SLIDING_WINDOW_SIZE*SLIDING_WINDOW_SIZE*3))
                 patch_data.append(padded)
                 mask_data.append([np.mean(mask[i:i+SLIDING_WINDOW_SIZE,j:j+SLIDING_WINDOW_SIZE]) > 0.5])
                 j+= 1
             i+= 1

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
    

    data = data.astype(np.float32)
    labels = np.squeeze(labels).astype(np.int32)
    print(data.shape, end = ", ")
    print(data.dtype)
    print(labels.shape, end = ", ")
    print(labels.dtype)
    np.savez(dataset_location, x = data, y = labels)
    
print("Loading data from {} ...".format(dataset_location))
ds = np.load(dataset_location)
data = ds['x']
labels = ds['y']
print("Data loaded. Start SVM optimization...")
# Train the SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(data, cv2.ml.ROW_SAMPLE, labels)
svm.save('./models/svm_weights_test.xml')
print("SVM training done.")


