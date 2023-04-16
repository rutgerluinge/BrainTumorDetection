from keras.optimizers import Adam

from image_load import split_data
from vit_keras import vit
from keras.layers import Layer


def start_procedure(data, labels, size=256, name="vit16l"):
    x, y, x_val, y_val, _, _ = split_data(data=data, label=labels)
    model = vit.vit_l16(image_size=(size, size),
                        activation="sigmoid",
                        include_top=True,
                        pretrained_top=False,
                        classes=2)

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy"])

    history = model.fit(x=x, y=y, epochs=10, validation_data=(x_val, y_val))

    model.save_weights(f"{name}_weights")
    model.save(f"saved_models/{name}", overwrite=True)
