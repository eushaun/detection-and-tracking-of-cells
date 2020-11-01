from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

import skimage.transform as trans
import matplotlib.pyplot as plt

def my_loss(y_true, y_pred):
    return - tf.reduce_sum(y_true * y_pred,
                           len(y_pred.get_shape()) - 1)

def jaccard_distance(weight):
    def loss(data, y_pred, smooth=100):
        
        intersection = K.sum(K.abs(y_pred * y_true * weight), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    return loss

def weighted_binary_crossentropy(data, y_pred):
#    print(len(data.shape))
    y_true = data[:, :256, ...]
    weight = data[:,256:, ...]
#    return -K.sum(y_true * K.log(y_pred))
    return K.binary_crossentropy(y_true, y_pred)*weight
#tf.reduce_sum(weight_map) + K.epsilon())

def dice_coef(data, y_pred, smooth=1):
    y_true = data[:, :256, ...]
    weight = data[:, 256:, ...]
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(data, y_pred):
    return 1-dice_coef(data, y_pred)

def unet(input_size = (256,256,1)):
    inputs = Input(input_size)
#weight_ip = Input(input_size)

    s = Lambda(lambda x: x / 255)(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c5)
    c5 = BatchNormalization()(c5)

    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    up6 = concatenate([up6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(up6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c6)
    c6 = BatchNormalization()(c6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    up7 = concatenate([up7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(up7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c7)
    c7 = BatchNormalization()(c7)

    up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    up8 = concatenate([up8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(up8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c8)
    c8 = BatchNormalization()(c8)

    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    up9 = concatenate([up9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(up9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(c9)
    c9 = BatchNormalization()(c9)

    c10 = Conv2D(1, (1, 1), activation='sigmoid')(c9)

#    _epsilon = convert_to_tensor(K.epsilon(), np.float32)
#    c11 = Lambda(lambda x: x / reduce_sum(x, len(x.get_shape()) - 1, True))(c10)
#    c11 = Lambda(lambda x: clip_by_value(x, _epsilon, 1. - _epsilon))(c11)
#    c11 = Lambda(lambda x: K.log(x))(c11)

    

#    weighted_sm = multiply([c11, weight_ip])

    model = Model(inputs=[inputs], outputs=[c10])
    
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy,
                  metrics=[dice_coef])

    K.set_value(model.optimizer.learning_rate, 0.001)

# model.summary()

    print(K.eval(model.optimizer.lr))

    return model

