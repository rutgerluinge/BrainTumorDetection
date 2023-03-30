# https://keras.io/guides/transfer_learning/
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras


def VGG_model():
    base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(256, 256, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

    base_model.trainable = False

    inputs = keras.Input((256, 256, 3))

    # base model running in inference mode
    x = base_model(inputs, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense classifier with one unit (binary classification)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="VGG")

    return model 
    
    

"""VGG CNN file"""
def start_procedure(train_data, validation_data):
    
    print("vgg algorithm here")

    model = VGG_model()
    model.summary()

    # Train model on new data
    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

    print("---------------Start fit (training)--------------------")
    model.fit(train_data, validation_data=validation_data, epochs=20)
    model.save_weights(filepath="./model_weights/VGG")
    model.save(filepath = Path("./models/VGG"), overwrite=True)

    print("---------------Start fine-tuning--------------------")
    model.trainable = True
    
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
    
    model.fit(train_data, validation_data=validation_data, epochs=10)
    model.save_weights(filepath="./model_weights/VGG")
    model.save(filepath = Path("./models/VGG"), overwrite=True)