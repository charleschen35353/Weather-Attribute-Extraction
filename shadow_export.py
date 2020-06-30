from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.enable_eager_execution()
from keras import backend as K
from os import path
from tensorflow import keras
from models import ShadowSeg
from dataloaders import MaskDataloader
from utils import freeze_session
from env import *

tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(tf.__version__)
#assert tf.executing_eagerly() == True

#create DNN model
testset_size = 638
data_path = "/home/charles/dataset/SBU-shadow/"
dl = MaskDataloader(data_path,test_exists = False)
_, _, test_generator = dl.load_dl()


batch_size = 16
checkpoint_path = "./cnn_ckpt/shadow_5.2/cp.ckpt"
checkpoint_dir = path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = ShadowSeg(input_shape = (IMG_HEIGHT, IMG_WIDTH, 1), regularization_factor = RF, learning_rate = 7e-4)

model.load_weights(checkpoint_path)
print("Weights Loaded")
loss, irbs = model.evaluate_generator(test_generator, steps = testset_size/batch_size,use_multiprocessing = False)
print("Restored model, loss: {}, error rate: {:5.2f}%".format(loss, irbs*100))

# Create, compile and train model...

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

print([out.op.name for out in model.outputs])
print([out.op.name for out in model.inputs])

tf.train.write_graph(frozen_graph, "./models", "model_shadow.pb", as_text=False)
[n.name for n in tf.get_default_graph().as_graph_def().node]

print("model output to pb successful.")

