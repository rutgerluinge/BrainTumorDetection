from pathlib import Path

from keras import Input
from keras.applications.densenet import layers
import tensorflow as tf
import numpy as np
# from vit_keras import vit
##create patches from original image
from keras.dtensor.optimizers import AdamW, Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

from image_load import split_data, data_augmentation


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_model(transformer_layers_count):
    """vit start"""
    patch_nr = 16
    total_amount_patches = patch_nr * patch_nr
    patch_size = 240 / patch_nr
    inputs = Input(shape=(240, 240, 3))
    # augmented = data_augmentation(inputs)     #without data_augmentation
    patches = Patches(patch_size)(inputs)

    encoded_patches = PatchEncoder(total_amount_patches, 64)(patches)

    num_heads = 4
    projection_dim = 64
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    mlp_head_units = [2048, 1024]
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers_count):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(2)(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


def gather_results(x_test, y_test, model):
    count = 0
    for idx, test_input in enumerate(x_test):
        y_hat = model.predict(test_input)

        if y_hat == y_test[idx]:
            count += 1


def start_procedure(x, y, x_val, y_val, transformer_layers=12, name="ViTL16"):
    vit_model = create_vit_model(transformer_layers)

    optimizer = AdamW(
        learning_rate=0.001, weight_decay=0.0001
    )

    vit_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    # no callbacks for now

    result = vit_model.fit(x=x, y=y,
                           batch_size=256,
                           epochs=100,
                           validation_data=(x_val, y_val))

    vit_model.save(f'{name}.h5')
    vit_model.save_weights(f'{name}_weights.h5')
    return vit_model

    # vit_model.save_weights(f"{name}_weights")
    # vit_model.save(f"{name}", overwrite=True)
