from tensorflow import keras
from tensorflow.keras import layers
from utils import IRBS

def CloudWindow(input_shape = (32,32,3), regularization_factor = 1e-2, learning_rate = 1e-4):
    input_img = layers.Input(shape = input_shape, dtype = 'float32', name = "input_img" ) 
    
    x = layers.Conv2D(16, 3, strides=(1, 1), name = "conv1",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(input_img)
    x = layers.Conv2D(32, 3, strides=(1, 1), name = "conv2",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(x)
    x = layers.Conv2D(64, 3, strides=(1, 1), name = "conv3",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(x)
    x = layers.Conv2D(64, 3, strides=(1, 1), name = "conv4",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation= 'relu', name = "fc", kernel_initializer = 'glorot_uniform')(x)
    output = layers.Dense(2, activation= 'softmax', name = "output", kernel_initializer = 'glorot_uniform')(x)
    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
              loss= keras.losses.categorical_crossentropy,
              metrics=['accuracy'] )
    return model

def CloudSegWindow(input_shape = (32,32,3), regularization_factor = 1e-2, learning_rate = 1e-4):
    input_img = layers.Input(shape = input_shape, dtype = 'float32', name = "input_img" ) 
    x = layers.Conv2D(16, 3, strides=(1, 1), name = "conv1",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(input_img)
    x = layers.Conv2D(32, 3, strides=(1, 1), name = "conv2",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(8*8*32, activation= 'relu', name = "fc", kernel_initializer = 'glorot_uniform')(x)
    x = layers.Reshape((8,8,32))(x)
    x = layers.Conv2D(32, 3, strides=(1, 1), name = "conv3",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(x)
    x = layers.Conv2D(16, 3, strides=(1, 1), name = "conv4",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(x)
    output = layers.Conv2D(2, 3, strides=(1, 1), name = "output",\
                                padding='same', activation="softmax", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(x)
    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
              loss= keras.losses.sparse_categorical_crossentropy,
              metrics=[IRBS] )
    return model


def SkySeg( input_shape = (256,256,3), regularization_factor = 1e-4, learning_rate = 7e-4): 
    
    input_img = layers.Input(shape = input_shape, dtype = 'float32', name = "input_img" ) 
    conv_1 = layers.Conv2D(16, 3, strides=(1, 1), name = "conv1",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(input_img)
  
    conv_2 = layers.Conv2D(16, 3, strides=(1, 1), name = "conv2",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform',  kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_1)
    pool_1 = layers.MaxPool2D()(conv_2)
    conv_3 = layers.Conv2D(32, 3, strides=(1, 1), name = "conv3",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_1)
    conv_4 = layers.Conv2D(32, 3, strides=(1, 1), name = "conv4",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_3)
    pool_2 = layers.MaxPool2D()(conv_4)
    conv_5 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv5",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_2)
    conv_6 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv6",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_5)
    pool_3 = layers.MaxPool2D()(conv_6)
    conv_7 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv7",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_3)
    conv_8 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv8",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_7)
    pool_4 = layers.MaxPool2D()(conv_8)
    conv_9 = layers.Conv2D(128, 3, strides=(1, 1), name = "conv9",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_4)
    conv_10 = layers.Conv2D(128, 3, strides=(1, 1), name = "conv10",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_9)
    pool_5 = layers.MaxPool2D()(conv_10)
    flat = layers.Flatten()(pool_5)
    fc_1 = layers.Dense(4*4*16, activation= 'relu', name = "fc_encode", kernel_initializer = 'glorot_uniform')(flat)
    fc_2 = layers.Dense(8*8*128, activation= 'relu', name = "fc_decode", kernel_initializer = 'glorot_uniform')(fc_1)
    rsp = layers.Reshape((8,8,128))(fc_2)
    up_1 = layers.UpSampling2D()(rsp)
    dcon_1 = layers.Conv2DTranspose(128, 3, strides = (1, 1), name = "dconv1",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_1)
    dcon_2 = layers.Conv2DTranspose(128, 3, strides = (1, 1), name = "dconv2",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_1)
    skip_1 = layers.Add()([conv_10,dcon_2])
    up_2 = layers.UpSampling2D()(skip_1)
    dcon_3 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv3",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_2)
    dcon_4 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv4",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_3)
    skip_2 = layers.Add()([conv_8,dcon_4])
    up_3 = layers.UpSampling2D()(skip_2)
    dcon_5 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv5",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_3)
    dcon_6 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv6",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_5)
    skip_3 = layers.Add()([conv_6,dcon_6])
    up_4 = layers.UpSampling2D()(skip_3)
    dcon_7 = layers.Conv2DTranspose(32, 3, strides = (1, 1), name = "dconv7",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_4)
    dcon_8 = layers.Conv2DTranspose(32, 3, strides = (1, 1), name = "dconv8",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_7)
    skip_4 = layers.Add()([conv_4,dcon_8])
    up_5 = layers.UpSampling2D()(skip_4)
    dcon_9 = layers.Conv2DTranspose(16, 3, strides = (1, 1), name = "dconv9",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_5)
    dcon_10 = layers.Conv2DTranspose(16, 3, strides = (1, 1), name = "dconv10",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_9)
    skip_5 = layers.Add()([conv_2,dcon_10])

    output = layers.Conv2D(2, 3, strides=(1, 1), name = "output",\
                                padding='same', activation="softmax", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(skip_5)
   
    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=[IRBS] )
    
    return model

def CloudSeg( input_shape = (256,256,3), regularization_factor = 1e-2, learning_rate = 1e-3): 
    
    input_img = layers.Input(shape = input_shape, dtype = 'float32', name = "input_img" ) 
    conv_1 = layers.Conv2D(16, 3, strides=(1, 1), name = "conv1",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(input_img)
  
    conv_2 = layers.Conv2D(16, 3, strides=(1, 1), name = "conv2",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform',  kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_1)
    pool_1 = layers.MaxPool2D()(conv_2)
    conv_3 = layers.Conv2D(32, 3, strides=(1, 1), name = "conv3",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_1)
    conv_4 = layers.Conv2D(32, 3, strides=(1, 1), name = "conv4",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_3)
    pool_2 = layers.MaxPool2D()(conv_4)
    conv_5 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv5",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_2)
    conv_6 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv6",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_5)
    pool_3 = layers.MaxPool2D()(conv_6)
    conv_7 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv7",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_3)
    conv_8 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv8",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_7)
    pool_4 = layers.MaxPool2D()(conv_8)
    conv_9 = layers.Conv2D(128, 3, strides=(1, 1), name = "conv9",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_4)
    conv_10 = layers.Conv2D(128, 3, strides=(1, 1), name = "conv10",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_9)
    pool_5 = layers.MaxPool2D()(conv_10)
    flat = layers.Flatten()(pool_5)
    fc_1 = layers.Dense(4*4*16, activation= 'relu', name = "fc_encode", kernel_initializer = 'glorot_uniform')(flat)
    fc_2 = layers.Dense(8*8*128, activation= 'relu', name = "fc_decode", kernel_initializer = 'glorot_uniform')(fc_1)
    rsp = layers.Reshape((8,8,128))(fc_2)
    up_1 = layers.UpSampling2D()(rsp)
    dcon_1 = layers.Conv2DTranspose(128, 3, strides = (1, 1), name = "dconv1",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_1)
    dcon_2 = layers.Conv2DTranspose(128, 3, strides = (1, 1), name = "dconv2",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_1)
    skip_1 = layers.Add()([conv_10,dcon_2])
    up_2 = layers.UpSampling2D()(skip_1)
    dcon_3 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv3",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_2)
    dcon_4 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv4",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_3)
    skip_2 = layers.Add()([conv_8,dcon_4])
    up_3 = layers.UpSampling2D()(skip_2)
    dcon_5 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv5",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_3)
    dcon_6 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv6",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_5)
    skip_3 = layers.Add()([conv_6,dcon_6])
    up_4 = layers.UpSampling2D()(skip_3)
    dcon_7 = layers.Conv2DTranspose(32, 3, strides = (1, 1), name = "dconv7",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_4)
    dcon_8 = layers.Conv2DTranspose(32, 3, strides = (1, 1), name = "dconv8",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_7)
    skip_4 = layers.Add()([conv_4,dcon_8])
    up_5 = layers.UpSampling2D()(skip_4)
    dcon_9 = layers.Conv2DTranspose(16, 3, strides = (1, 1), name = "dconv9",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_5)
    dcon_10 = layers.Conv2DTranspose(16, 3, strides = (1, 1), name = "dconv10",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_9)
    skip_5 = layers.Add()([conv_2,dcon_10])

    output = layers.Conv2D(2, 3, strides=(1, 1), name = "output",\
                                padding='same', activation="softmax", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(skip_5)
   
    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=[IRBS] )
    
    return model


def ShadowSeg( input_shape = (256,256,1), regularization_factor = 1e-4, learning_rate = 7e-4): 
    
    input_img = layers.Input(shape = input_shape, dtype = 'float32', name = "input_img" ) 
    conv_1 = layers.Conv2D(16, 3, strides=(1, 1), name = "conv1",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(input_img)
  
    conv_2 = layers.Conv2D(16, 3, strides=(1, 1), name = "conv2",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform',  kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_1)
    pool_1 = layers.MaxPool2D()(conv_2)
    conv_3 = layers.Conv2D(32, 3, strides=(1, 1), name = "conv3",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_1)
    conv_4 = layers.Conv2D(32, 3, strides=(1, 1), name = "conv4",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_3)
    pool_2 = layers.MaxPool2D()(conv_4)
    conv_5 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv5",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_2)
    conv_6 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv6",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_5)
    pool_3 = layers.MaxPool2D()(conv_6)
    conv_7 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv7",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_3)
    conv_8 = layers.Conv2D(64, 3, strides=(1, 1), name = "conv8",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_7)
    pool_4 = layers.MaxPool2D()(conv_8)
    conv_9 = layers.Conv2D(128, 3, strides=(1, 1), name = "conv9",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(pool_4)
    conv_10 = layers.Conv2D(128, 3, strides=(1, 1), name = "conv10",\
                                padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(conv_9)
    pool_5 = layers.MaxPool2D()(conv_10)
    flat = layers.Flatten()(pool_5)
    fc_1 = layers.Dense(4*4*16, activation= 'relu', name = "fc_encode", kernel_initializer = 'glorot_uniform')(flat)
    fc_2 = layers.Dense(8*8*128, activation= 'relu', name = "fc_decode", kernel_initializer = 'glorot_uniform')(fc_1)
    rsp = layers.Reshape((8,8,128))(fc_2)
    up_1 = layers.UpSampling2D()(rsp)
    dcon_1 = layers.Conv2DTranspose(128, 3, strides = (1, 1), name = "dconv1",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_1)
    dcon_2 = layers.Conv2DTranspose(128, 3, strides = (1, 1), name = "dconv2",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_1)
    skip_1 = layers.Add()([conv_10,dcon_2])
    up_2 = layers.UpSampling2D()(skip_1)
    dcon_3 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv3",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_2)
    dcon_4 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv4",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_3)
    skip_2 = layers.Add()([conv_8,dcon_4])
    up_3 = layers.UpSampling2D()(skip_2)
    dcon_5 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv5",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_3)
    dcon_6 = layers.Conv2DTranspose(64, 3, strides = (1, 1), name = "dconv6",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_5)
    skip_3 = layers.Add()([conv_6,dcon_6])
    up_4 = layers.UpSampling2D()(skip_3)
    dcon_7 = layers.Conv2DTranspose(32, 3, strides = (1, 1), name = "dconv7",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_4)
    dcon_8 = layers.Conv2DTranspose(32, 3, strides = (1, 1), name = "dconv8",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_7)
    skip_4 = layers.Add()([conv_4,dcon_8])
    up_5 = layers.UpSampling2D()(skip_4)
    dcon_9 = layers.Conv2DTranspose(16, 3, strides = (1, 1), name = "dconv9",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(up_5)
    dcon_10 = layers.Conv2DTranspose(16, 3, strides = (1, 1), name = "dconv10",\
                                       padding='same', activation="relu", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(dcon_9)
    skip_5 = layers.Add()([conv_2,dcon_10])

    output = layers.Conv2D(2, 3, strides=(1, 1), name = "output",\
                                padding='same', activation="softmax", kernel_initializer='glorot_uniform', kernel_regularizer = keras.regularizers.l2(regularization_factor))(skip_5)
   
    model = keras.Model(inputs=[input_img], outputs=output)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=[IRBS] )
    
    return model
