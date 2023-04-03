from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np
import os
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image



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

def load_images_method_2():
    folder_path = "brain_tumor_dataset"
    no_images = os.listdir(folder_path + '/no/')
    yes_images = os.listdir(folder_path + '/yes/')
    dataset = []
    labels = []

    for image_name in no_images:
        image = cv2.imread(folder_path + '/no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((240, 240))
        dataset.append(np.array(image))
        labels.append(0)

    for image_name in yes_images:
        image = cv2.imread(folder_path + '/yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((240, 240))
        dataset.append(np.array(image))
        labels.append(1)

    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels

def split_data(data, label):
    _70_idx = int(len(data) * 0.7)
    _90_idx = int(len(data) * 0.9)

    x_train = data[:_70_idx]
    y_train = label[:_70_idx]

    x_validate = data[_70_idx:_90_idx]
    y_validate = label[_70_idx:_90_idx]

    return np.array(x_train), np.array(y_train), np.array(x_validate), np.array(y_validate)

if __name__ == '__main__':
    train, test = load_images()
