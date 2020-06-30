import glob
import argparse
import tensorflow as tf
import cv2
import numpy as np
import argparse
from os import path, makedirs
from env import *
from utils import pb_predict_mask, load_test_image, load_mask, pct_to_label, R_squared, visualize, rgb2gray
tf.logging.set_verbosity(tf.logging.ERROR)

model_path = "./models/model_shadow.pb"

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
    img_gray = np.expand_dims(rgb2gray(img), axis = -1)
    mask = pb_predict_mask(sess, img_gray) < 0.3
    mask_pct = np.sum(mask)/(mask.shape[0]*mask.shape[1])*100
    gt_mask = load_mask(mask_path,  dsize = (IMG_HEIGHT, IMG_WIDTH))/255.0 > 0.5
    gt_mask_pct = np.sum(gt_mask)/(gt_mask.shape[0]*gt_mask.shape[1])*100
    y_pred.append(mask_pct)
    y_true.append(gt_mask_pct)
    
    if pct_to_label(mask_pct) != pct_to_label(gt_mask_pct):
        wrong_count +=1
        print("Wrong Prediction. {} has around {}% of sky area. Predicted:{}%".format(img_path, PCT_LVL[pct_to_label(gt_mask_pct)],  PCT_LVL[pct_to_label(mask_pct)]))
        print("More info on unsampled pct. GT:{}% Predicted:{}%".format(gt_mask_pct, mask_pct))
    
    if not path.exists('./results_shadow'):
        makedirs('./results_shadow')
    visualize(img, mask, path = "./results_shadow/"+img_path.split("/")[-1].split(".")[0]+"_pred.jpg" )
    j+=1
    

y_pred = np.array(y_pred)
y_true = np.array(y_true)
r2 = R_squared(y_true,y_pred)
print("R squared: {}".format(r2))
print("Accuracy: {}%".format( 1- 1.0* wrong_count / j))
