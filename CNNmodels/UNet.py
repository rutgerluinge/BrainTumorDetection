from pathlib import Path

from keras import Input, Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Rescaling, Flatten, Dense, Activation
from keras.optimizers import Adam

def start_procedure(train_data, validation_data):

    model = Sequential([Unet_model(),                   #original U-Net model
                        Flatten(),                      #flatten layer to 1d
                        Dense(1, activation='sigmoid')  #add 1 fully connected layer to output node
                        ])
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.1), loss="binary_crossentropy", metrics=['binary_accuracy'])

    print("---------------Start fit (training)--------------------")
    model.fit(train_data, validation_data=validation_data, epochs=100)
    model.save_weights(filepath="model_weights")
    model.save(filepath=Path("brain_tumor_dataset"), overwrite=True)


def Unet_model():

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

    output = Conv2D(1, 1, activation='sigmoid', name="output")(d1)

    return Model(inputs, output, name="U-Net")

