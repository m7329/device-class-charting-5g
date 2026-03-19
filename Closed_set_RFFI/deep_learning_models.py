from keras.layers import Input, Lambda, ReLU, Add
from keras.models import Model
from keras.layers import (Dense, Conv2D, Flatten, BatchNormalization, AveragePooling2D)
from keras.saving import register_keras_serializable

import numpy as np
import tensorflow as tf

# In[]
'''Residual block'''
def resblock(x, kernelsize, filters, first_layer=False):
    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        fx = BatchNormalization()(fx)

        x = Conv2D(filters, 1, padding='same')(x)

        out = Add()([x, fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        fx = BatchNormalization()(fx)
        # 
        out = Add()([x, fx])
        out = ReLU()(out)

    return out


# Register Lambda layer for (de-)serialization
@register_keras_serializable(package="rffi")
def l2_normalize_axis1(x):
    return tf.nn.l2_normalize(x, axis=1)


def classification_net(datashape, num_classes):
    datashape = datashape

    inputs = Input(shape=datashape[1:])

    x = Conv2D(32, 7, strides=2, activation='relu', padding='same')(inputs)

    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)

    x = resblock(x, 3, 64, first_layer=True)
    x = resblock(x, 3, 64)

    # PCA can reduce the input spatial dims (e.g., H=2, W=3), which after the stride-2
    # conv/resblocks can lead to H=1. Using padding="same" avoids negative-dimension
    # errors for small feature maps while keeping the pooling behavior similar.
    x = AveragePooling2D(pool_size=2, padding="same")(x)

    x = Flatten()(x)

    x = Dense(512)(x)

    # Keras 3 requires explicit output_shape for Lambda layers
    x = Lambda(l2_normalize_axis1, output_shape=lambda s: s, name='feature_layer')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
