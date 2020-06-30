import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import matplotlib.image as pltimg
from tensorflow.python.platform import gfile
from env import PCT_LVL

def one_hot_encode(labels): # zero base
    return (np.arange(labels.max()+1) == labels[...,None]).astype(int)
    

def load_test_image(img_path, dsize = (256,256)):
    img = pltimg.imread(img_path)
    if np.mean(img) < 1:
        img = (img*255).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis = 2)
    elif len(img.shape) > 3:
        print("Bad shape.")

    if img.shape[-1] == 1 :
        img = np.tile(img, (1,1,3))
    elif img.shape[-1] == 4:
        img = img[:,:,:3]
    elif img.shape[-1] == 3: 	
        pass
    else:
        print("Bad channel {}".format(img_path))

    img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    return img

def load_mask(mask_path, dsize = (256,256)):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if np.mean(img) < 1:
        img = (img*255).astype(np.uint8)
    img = cv2.resize(img, dsize = dsize, interpolation=cv2.INTER_CUBIC)
    return img

def pct_to_label(pct):
    label = -1
    if pct <= int(PCT_LVL[0]): label = 0
    elif pct <= int(PCT_LVL[1]): label = 1
    elif pct <= int(PCT_LVL[2]): label = 2
    else: label = 3
    return label

def rgb2hsv(rgb, normalization = False):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    
    if normalization: # use histogram equalization
        img_int = (hsv[...,2]*255).astype(np.uint8)
        if len(img_int.shape) == 2:
            img_int = cv2.equalizeHist(img_int)
        else:
            temp = img_int
            img_int = []
            for t in temp:
                img_int.append(cv2.equalizeHist(t))
            img_int = np.array(img_int)
        hsv[...,2] = img_int/255

    return hsv

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def rgb_apply_mask(rgb_img, bi_mask):
    return rgb_img * (bi_mask[:, :, None] * np.ones(3, dtype=int)[None, None, :])

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
  
    return loss

def IRBS(y_true, y_pred, from_logits = False): 
    if from_logits: 
        print("IRBS not implemented. Return None.")
        return None
    true_mask = tf.cast(y_true, tf.int32) # sample y_true
    pred_mask = tf.cast(tf.argmax(y_pred, axis = -1), tf.int32) #sample y_pred
    diff = tf.math.abs(true_mask - pred_mask) # calc difference of preidction
    diff = tf.reshape(diff, (-1, tf.shape(y_true)[1] * tf.shape(y_true)[2]))
    error_rates = (tf.reduce_sum(diff, axis = -1) / (tf.shape(y_true)[1] * tf.shape(y_true)[2])) 
    mean_error_rate = tf.reduce_mean(error_rates)
    return mean_error_rate

def pb_predict_window_mask(sess, img, window_size = 32): #image in hsv scale 0-1
    data = []
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            patch = img[i:i+window_size,j:j+window_size,:]
            padded = np.zeros(shape = ( window_size, window_size ,3 ) )
            padded[:patch.shape[0],:patch.shape[1]] = patch
            padded = np.reshape(padded, (window_size,window_size,3))
            data.append(padded)
            j+= 1
        i += 1
    data = np.array(data).astype(np.float32)
    output_tensor = sess.graph.get_tensor_by_name('import/output/Softmax:0')
    y_pred = sess.run(output_tensor, {'import/input_img:0':data}) 
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.reshape(y_pred, (img.shape[0], img.shape[1]))
    mask = np.zeros( shape = (img.shape[0], img.shape[1]) )
    weights = np.zeros( shape = (img.shape[0], img.shape[1]))
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            mask[i:i+window_size,j:j+window_size] += y_pred[i,j]
            weights[i:i+window_size,j:j+window_size] += 1
            j+= 1
        i += 1
    
    mask = mask / weights
    return mask

def pb_predict_window_seg(sess, img, window_size = 32): #image in hsv scale 0-1
    data = []
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            patch = img[i:i+window_size,j:j+window_size,:]
            padded = np.zeros(shape = ( window_size, window_size ,3 ) )
            padded[:patch.shape[0],:patch.shape[1]] = patch
            padded = np.reshape(padded, (window_size,window_size,3))
            data.append(padded)
            j+= 1
        i += 1
    data = np.array(data).astype(np.float32)
    output_tensor = sess.graph.get_tensor_by_name('import/output/truediv:0')
    y_pred = sess.run(output_tensor, {'import/input_img:0':data}) 
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = np.reshape(y_pred, (img.shape[0], img.shape[1], window_size, window_size))
    mask = np.zeros( shape = (img.shape[0]+window_size, img.shape[1]+window_size) )
    weights = np.zeros( shape = (img.shape[0]+window_size, img.shape[1]+window_size))
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            mask[i:i+window_size,j:j+window_size] += y_pred[i,j,:,:]
            weights[i:i+window_size,j:j+window_size] += 1
            j+= 1
        i += 1
    
    mask = mask / weights
    mask = mask[:img.shape[0], :img.shape[1]]
    return mask

def pb_predict_mask(sess, image):
    image = image*1.0/255.0 # np image
    image = np.expand_dims(image, axis = 0)
    output_tensor = sess.graph.get_tensor_by_name('import/output/truediv:0')
    predictions = sess.run(output_tensor, {'import/input_img:0':image})
    return predictions[0,:,:,0] # 512 * 512

def pb_predict_label(sess, image):
    image = image*1.0/255.0 # np image
    image = np.expand_dims(image, axis = 0) #np.resize(img, [1,256, 256, 3])
    output_tensor = sess.graph.get_tensor_by_name('import/output/Softmax:0')
    predictions = sess.run(output_tensor, {'import/input_img:0':image})
    return predictions[0] # 1

def svm_predict_window_mask(svm, img, window_size = 8):
    
    data = []
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            patch = img[i:i+window_size,j:j+window_size,:]
            padded = np.zeros(shape = ( window_size, window_size ,3 ) )
            padded[:patch.shape[0],:patch.shape[1]] = patch
            padded = np.reshape(padded, (window_size*window_size*3))
            data.append(padded)
            j+= 1
        i += 1
    data = np.array(data).astype(np.float32)
    _, y_pred = svm.predict(data) 
    y_pred = np.reshape(y_pred, (img.shape[0], img.shape[1]))
    mask = np.zeros( shape = (img.shape[0], img.shape[1]) )
    weights = np.zeros( shape = (img.shape[0], img.shape[1]))
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            mask[i:i+window_size,j:j+window_size] += y_pred[i,j]
            weights[i:i+window_size,j:j+window_size] += 1
            j+= 1
        i += 1
    mask = mask / weights
    return mask

def R_squared(y, y_pred):
    residual = np.sum(np.square(np.subtract(y, y_pred)))
    total = np.sum(np.square(np.subtract(y, np.mean(y))))
    r2 = np.subtract(1.0, np.divide(residual, total))
    return r2

def visualize(img, mask, path = "./"):
    img_vis = img
    mask_vis = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]*255
    temp_1 = np.concatenate((img_vis, mask_vis), axis = 1)
    temp_2 = np.concatenate((img_vis * (mask_vis > 0 ), img_vis * (mask_vis == 0 )), axis=1)
    vis = np.concatenate((temp_1, temp_2), axis=0).astype(np.uint8)
    
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR) )
