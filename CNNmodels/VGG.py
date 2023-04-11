# https://keras.io/guides/transfer_learning/ # TODO: change optimizer to SGD with learning rate schedule 
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from image_load import split_data


def VGG_model():
    base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

    base_model.trainable = False

    inputs = keras.Input((224, 224, 3))

    # base model running in inference mode
    x = base_model(inputs, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense classifier with one unit (binary classification)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="VGG")

    return model 
    
    

"""VGG CNN file"""
def start_procedure(data, labels):
    # data
    x, y, x_val, y_val, _, _ = split_data(data=data, label=labels)
    
    print("vgg algorithm here")

    model = VGG_model()
    model.summary()

    # Train model on new data
    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

    print("---------------Start fit (training)--------------------")
    model.fit(x=x, y=y, validation_data=(x_val,y_val), epochs=20)
    model.save_weights(filepath="../model_weights/VGG/")
    model.save(filepath = Path("../models/VGG/"), overwrite=True)

    print("---------------Start fine-tuning--------------------")
    # load model from path, comment out if not needed
    model = keras.models.load_model(filepath = Path("../models/VGG/"))
    
    model.trainable = True
    
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
    
    model.fit(x=x, y=y, validation_data=(x_val,y_val), epochs=10)
    model.save_weights(filepath="../model_weights/VGG/")
    model.save(filepath = Path("../models/VGG/"), overwrite=False)