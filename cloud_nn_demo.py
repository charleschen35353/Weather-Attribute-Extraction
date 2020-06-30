import glob
import argparse
import cv2
import numpy as np
import argparse
import tensorflow as tf
from os import path, makedirs
from env import *
from utils import load_test_image, load_mask, rgb2hsv, pb_predict_window_seg, pct_to_label, R_squared, visualize
model_path = "./models/model_cloud_seg.pb"

with tf.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

sess = tf.Session(graph=graph)


parser = argparse.ArgumentParser()
parser.add_argument("target_ds")
args = parser.parse_args()

print("model loaded.")

d_img = path.join(args.target_ds,"images")+"/*"
d_mask = path.join(args.target_ds, "masks")+"/*"
img_paths = sorted(glob.glob(d_img))
mask_paths = sorted(glob.glob(d_mask))


j, wrong_count = 0,0
y_pred, y_true = [], []

for img_path, mask_path in zip(img_paths, mask_paths):
    img = load_test_image(img_path, dsize = (IMG_HEIGHT, IMG_WIDTH))
    img_hsv = rgb2hsv(img, normalization = True) 
    mask = pb_predict_window_seg(sess, img_hsv, window_size = SLIDING_WINDOW_SIZE) > CLOUD_THRES
    #mask = pb_predict_window_mask(sess, img_hsv, window_size = SLIDING_WINDOW_SIZE) > CLOUD_THRES
    #mask = np.zeros_like(img_hsv[...,0])
    mask_pct = np.sum(mask)/(mask.shape[0]*mask.shape[1])*100
    gt_mask = load_mask(mask_path,  dsize = (IMG_HEIGHT, IMG_WIDTH))/255.0 > CLOUD_THRES
    gt_mask_pct = np.sum(gt_mask)/(gt_mask.shape[0]*gt_mask.shape[1])*100
    y_pred.append(mask_pct)
    y_true.append(gt_mask_pct)
    
    if pct_to_label(mask_pct) != pct_to_label(gt_mask_pct):
        wrong_count +=1
        print("Wrong Prediction. {} has around {}% of sky area. Predicted:{}%".format(img_path, PCT_LVL[pct_to_label(gt_mask_pct)],  PCT_LVL[pct_to_label(mask_pct)]))
        print("More info on unsampled pct. GT:{}% Predicted:{}%".format(gt_mask_pct, mask_pct))
    
    if not path.exists('./results_cloud'):
        makedirs('./results_cloud')
    visualize(img, mask, path = "./results_cloud/"+img_path.split("/")[-1].split(".")[0]+"_pred.jpg" )
    j+=1
