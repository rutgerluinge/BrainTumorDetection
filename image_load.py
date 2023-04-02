from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np
import pandas as pd
from pathlib import Path


def load_images(width=244,height=244) -> {DirectoryIterator, DirectoryIterator}:

    batch_size: int = 32

    data_directory = Path("brain_tumor_dataset")
    print(data_directory)

    data_generator = ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=0.2,
    )

    train_data_set = data_generator.flow_from_directory(
        directory=data_directory,
        batch_size=batch_size,
        subset="training",
        shuffle=True,
        class_mode="binary",
        target_size=(height, width),
        classes={'no': 0, 'yes': 1}
    )

    validation_data_set = data_generator.flow_from_directory(
        directory=data_directory,
        batch_size=batch_size,
        subset="validation",
        shuffle=True,
        class_mode="binary",
        target_size=(height, width),
        classes={'no': 0., 'yes': 1.}
    )

    return train_data_set, validation_data_set

def split_input_label(dataset):
    x = []
    y = []
    for x_batch, y_batch in dataset:
        x.append(x_batch)
        y.append(y_batch)

    return x, y


if __name__ == '__main__':
    train, test = load_images()
