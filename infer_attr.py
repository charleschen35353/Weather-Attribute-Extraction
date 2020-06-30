import glob
import argparse
import tensorflow as tf
import cv2
import numpy as np
import argparse
from PIL import ImageFont, ImageDraw, Image
from os import path, makedirs
from env import *
from utils import *
tf.logging.set_verbosity(tf.logging.ERROR)

model_sky_path = "./models/model_sky_new.pb"
model_sun_path = "./models/model_sunny.pb"
model_cloud_path = "./models/model_cloud_seg.pb"
model_rain_path = "./models/model_rain.pb"
model_shadow_path = "./models/model_shadow.pb"

with tf.gfile.GFile(model_sky_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph_skyseg:
    tf.import_graph_def(graph_def)

with tf.gfile.GFile(model_sun_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph_sunny:
    tf.import_graph_def(graph_def)

with tf.gfile.GFile(model_cloud_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph_cloud:
    tf.import_graph_def(graph_def)

with tf.gfile.GFile(model_rain_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph_rain:
    tf.import_graph_def(graph_def)

with tf.gfile.GFile(model_shadow_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph_shadow:
    tf.import_graph_def(graph_def)

sky_seg_model = tf.Session(graph=graph_skyseg)
sunny_model = tf.Session(graph=graph_sunny)  
cloud_model = tf.Session(graph=graph_cloud) 
rain_model = tf.Session(graph=graph_rain) 
shadow_model = tf.Session(graph=graph_shadow) 

parser = argparse.ArgumentParser()
parser.add_argument("target_ds")
args = parser.parse_args()
print("model loaded.")

d_img = path.join(args.target_ds)+"/*"
img_paths = sorted(glob.glob(d_img))
if not path.exists('./results'):
        makedirs('./results') 
if DEBUG:
    if not path.exists('./vis'):
            makedirs('./vis')

for img_path in img_paths:
    img = load_test_image(img_path, dsize = (IMG_HEIGHT, IMG_WIDTH))

    #predict sky area
    sky_mask = pb_predict_mask(sky_seg_model, img) < SKY_THRES
    sky_pct = np.sum(sky_mask)/(IMG_HEIGHT*IMG_WIDTH)*100
    visualize(img, sky_mask, path = "./vis/"+img_path.split("/")[-1].split(".")[0]+"_sky_pred.jpg" )
    
    #predict cloud area
    cloud_pct = -1
    if sky_pct < 1:
        cloud_pct = 0.0
    else:
        img_hsv = rgb2hsv(img, normalization = True)
        cloud_mask = pb_predict_window_seg(cloud_model, img_hsv, window_size = SLIDING_WINDOW_SIZE) > CLOUD_THRES
        cloud_mask = cloud_mask * sky_mask
        cloud_pct = np.sum(cloud_mask)/np.sum(sky_mask)*100
        visualize(img, cloud_mask, path = "./vis/"+img_path.split("/")[-1].split(".")[0]+"_cloud_pred.jpg" )

    #non sky area prediction
    land_mask = 1 -sky_mask
    land = rgb_apply_mask(img, land_mask)
    #sun light estmiation
    pass
    #rain estimation
    img512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    rain_logits = pb_predict_label(rain_model, img512)
    rain_pct =  100*(rain_logits[0] + rain_logits[1])
    #water estimation
    pass
    #shadow estimation
    contrast_ind = 0
    img_gray = np.expand_dims(rgb2gray(img), axis = -1)
    shadow_mask = pb_predict_mask(shadow_model, img_gray) < SHADOW_THRES 
    shadow_pct = np.sum(shadow_mask)/(IMG_HEIGHT*IMG_WIDTH)*100
    if shadow_pct > 0.5:
        mask_diff = (cv2.dilate((shadow_mask*255).astype(np.uint8), np.ones((11, 11)), iterations = 2) > 0) != shadow_mask
        avg_shadow_intensity = rgb2hsv(rgb_apply_mask(img, shadow_mask))[...,2]/255.0
        avg_shadow_intensity = np.sort(avg_shadow_intensity[avg_shadow_intensity!=0])
        avg_shadow_intensity = np.mean(avg_shadow_intensity[len(avg_shadow_intensity)//4 : len(avg_shadow_intensity)//4*3])
        avg_margin_intensity = rgb2hsv(rgb_apply_mask(img, mask_diff))[...,2]/255.0
        avg_margin_intensity = np.sort(avg_margin_intensity[avg_margin_intensity!=0])
        avg_margin_intensity = np.mean(avg_margin_intensity[len(avg_margin_intensity)//4 : len(avg_margin_intensity)//4*3])
        if avg_margin_intensity/avg_shadow_intensity <= 4 : contrast_ind = 1
        else: contrast_ind = 2
        #vis = np.concatenate((rgb_apply_mask(img, shadow_mask), rgb_apply_mask(img, mask_diff)), axis = 1)
        #cv2.imwrite("./vis/"+img_path.split("/")[-1].split(".")[0]+"_diff_pred.jpg", vis )
        #visualize(img, shadow_mask, path = "./vis/"+img_path.split("/")[-1].split(".")[0]+"_shadow_pred.jpg" )
    
 
    #sunny classification
    sunny_pct = pb_predict_label(sunny_model, img)[1]*100

    text_canvas = np.zeros_like(img)

    #"Image {} weather attribute analysis:".format(img_path) + "\n"\  
    info = "Weather attribute analysis:" + "\n"\
          +"Sky area (over entire image): {:.2f}%".format(sky_pct) + "\n"\
          +"Cloud area (over sky area): {:.2f}%".format(cloud_pct)+ "\n"\
          +"Sunniness: {:.2f}%".format(sunny_pct)+ "\n"\
          +"Rain probability: {:.2f}%".format(rain_pct)+ "\n" \
          +"Shadow contrast level:{}".format(CONTRAST_LEVEL[contrast_ind])

    pil_im = Image.fromarray(text_canvas)
    draw = ImageDraw.Draw(pil_im)
    draw.text((20, 50), info, font=ImageFont.load_default())
    text_canvas = np.array(pil_im)
    
    img = cv2.cvtColor(np.concatenate((img, text_canvas), axis = 1) , cv2.COLOR_RGB2BGR)
    cv2.imwrite("./results/"+img_path.split("/")[-1].split(".")[0]+"_result.jpg", img)
