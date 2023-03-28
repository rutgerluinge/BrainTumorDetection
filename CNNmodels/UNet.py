from pathlib import Path
from typing import List

from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Rescaling, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras.applications import *
import matplotlib.pyplot as plt
import numpy as np
from image_load import split_input_label
from keras import backend
def thresholded_relu(theta):
    def _thresholded_relu(x):
        return backend.cast(backend.greater(x, theta), dtype='float32')
    return _thresholded_relu

def start_procedure(train_data, validation_data):
    #model = get_U_Net_model()
    model = Unet_model()
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['binary_accuracy'])

    print("---------------Start fit (training)--------------------")
    model.fit(train_data, validation_data=validation_data, epochs=30)
    model.save_weights(filepath="model_weights")
    model.save(filepath=Path("brain_tumor_dataset"), overwrite=True)


def Unet_model():
    thresholded_relu_activation  = Activation(thresholded_relu(0.5))
    def encode(input_layer, filters):
        """2 convolutions + 1 max pool (2x2)"""
        conv_1 = Conv2D(filters, 3, activation="relu", padding="same")(input_layer)
        conv_2 = Conv2D(filters, 3, activation="relu", padding="same")(conv_1)
        max_pool = MaxPooling2D(pool_size=(2, 2))(conv_2)

        return max_pool, conv_2

    def decoder(input_layer, filters, concat_layer):
        """up conv, concat, 2 convolutions"""
        conv_1 = Conv2D(filters, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(input_layer))
        concat = concatenate([concat_layer, conv_1], axis=-1)  # concatenate 2 layers
        conv_2 = Conv2D(filters, 3, activation="relu", padding="same")(concat)
        conv_3 = Conv2D(filters, 3, activation="relu", padding="same")(conv_2)

        return conv_3

    inputs = Input((256, 256, 3))
    pool1, e1 = encode(inputs, 64)
    pool2, e2 = encode(pool1, 128)
    pool3, e3 = encode(pool2, 256)
    pool4, e4 = encode(pool3, 512)

    base = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
    base = Conv2D(1024, 3, activation="relu", padding="same")(base)

    d4 = decoder(base, 512, e4)
    d3 = decoder(d4, 256, e3)
    d2 = decoder(d3, 128, e2)
    d1 = decoder(d2, 64, e1)

    output = Conv2D(1, 1, activation='relu', name="output")(d1)
    # segmentation_map = Conv2D(2, 3, activation="relu", padding="same")(d1)
    # output = Conv2D(1, 1, activation="sigmoid")(segmentation_map)
    x = Flatten()(output)  #added this myself, as
    output = Dense(1, activation="sigmoid")(x)  # not sure if this is necessary however shape was not equal

    return Model(inputs, output, name="U-Net")

