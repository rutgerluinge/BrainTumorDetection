# https://github.com/faustomorales/vit-keras

import numpy as np
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize
from keras.optimizers import Adam
from image_load import split_data


def start_procedure(data, labels, size = 224, name = "vit16b"):
    x, y, x_val, y_val, _, _ = split_data(data=data, label=labels)

    model = vit.vit_l16(image_size=(size, size),
                        activation='sigmoid',
                        pretrained=True,
                        include_top=True,
                        pretrained_top=False,
                        classes=2)
    classes = [0, 1]

    model.compile(optimizer=Adam(),
                      loss="binary_crossentropy",
                      metrics=["binary_accuracy"])

    history = model.fit(x=x, y=y, epochs=100, validation_data=(x_val,y_val))

    model.save_weights(f"{name}_weights")
    model.save(f"saved_models/{name}", overwrite=True)


    # Get an image and compute the attention map
    image = "./brain_tumor_dataset/yes/Y7.jpg"
    attention_map = visualize.attention_map(model=model, image=image)
    print('Prediction:', classes[
        model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
    )  

    # Plot results
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(image)
    _ = ax2.imshow(attention_map)