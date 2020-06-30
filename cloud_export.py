from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.enable_eager_execution()
from keras import backend as K
from os import path
from tensorflow import keras
from models import CloudSeg
from dataloaders import *
from utils import freeze_session
from env import *
tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(tf.__version__)
#assert tf.executing_eagerly() == True

batch_size = 16
epochs = 200
checkpoint_path = "./cnn_ckpt/cloud3/cp.ckpt"
data_path = "/home/charles/dataset/swimseg/"
checkpoint_dir = path.dirname(checkpoint_path)
dl = MaskDataloader(data_path, test_exists = False)
_, _, test_dataset = dl.load_dl()
test_size = 93
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model = CloudSeg()
print(model.summary())
model.load_weights(checkpoint_path)
print("Weights loaded")
loss, acc = model.evaluate_generator(test_dataset, steps = test_size/batch_size,use_multiprocessing = False)
print("Restored model, loss: {}, IRBS: {:5.2f}%".format(loss, acc*100))

# Create, compile and train model...

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

print([out.op.name for out in model.outputs])
print([out.op.name for out in model.inputs])

tf.train.write_graph(frozen_graph, "./models", "model_cloud_seg_e2e.pb", as_text=False)
[n.name for n in tf.get_default_graph().as_graph_def().node]

print("model output to pb successful.")

