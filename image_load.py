from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np
import os
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from copy import copy

def load_images(width=256, height=256) -> {DirectoryIterator, DirectoryIterator}:
    batch_size: int = 32

    data_directory = Path("brain_tumor_dataset")

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


def load_images_method_2(image_input_size=242):
    """simple to understand method to load the data into 2 lists (np.array): data, labels"""
    folder_path = "brain_tumor_dataset"
    no_images = os.listdir(folder_path + '/no/')
    yes_images = os.listdir(folder_path + '/yes/')
    dataset = []
    labels = []

    for image_name in no_images:
        image = cv2.imread(folder_path + '/no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image_input_size, image_input_size))
        dataset.append(np.array(image))
        labels.append(0)

    for image_name in yes_images:
        image = cv2.imread(folder_path + '/yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((image_input_size, image_input_size))
        dataset.append(np.array(image))
        labels.append(1)

    return np.array(dataset), np.array(labels)


def split_data(data, label):
    """ split data into train/validation/test data:
        70/20/10
        @TODO currently only returns train and validation sets!!!!"""
    _70_idx = int(len(data) * 0.7)
    _90_idx = int(len(data) * 0.9)

    x_train = data[:_70_idx]
    y_train = label[:_70_idx]

    x_validate = data[_70_idx:_90_idx]
    y_validate = label[_70_idx:_90_idx]

    x_test = data[_90_idx:]
    y_test = label[_90_idx:]

    return np.array(x_train), np.array(y_train), np.array(x_validate), np.array(y_validate), x_test, y_test


def shuffle_data(data, labels):
    """shuffle data whilst remaining the correct label indices."""
    joined_lists = list(zip(data, labels))
    np.random.shuffle(joined_lists)  # Shuffle "joined_lists" in place
    data, labels = zip(*joined_lists)
    return   list(data), list(labels)


def data_augmentation(data_images, labels):
    """ @data: input data in array form (np.array).
        @label: data labels (np.array) which corresponds to the data indexes.
        @:returns new data and labels with more data (augmented) by rotation.
        @: rotation/mirror
    """
    seq = iaa.Sequential([
        iaa.Flipud(p=0.5),  # flip the image vertically with probability 0.5
        iaa.Affine(rotate=(-10, 10)),  # rotate the image by -10 to 10 degrees
        iaa.GaussianBlur(sigma=(0, 1.0)),  # blur the image with a sigma of 0 to 1.0
    ])
    plt.fig, axes = plt.subplots(nrows=1, ncols=2)

    # for image in data_images:     #uncomment to see
    #     augmented_image = seq(image=image)
    #
    #     axes[0].imshow(image, cmap='gray')
    #     axes[0].set_title('original')
    #
    #     # Plot the second image on the second subplot
    #     axes[1].imshow(augmented_image, cmap='gray')
    #     axes[1].set_title('augmented')
    #     plt.show()

    for image, label in zip(copy(data_images), copy(labels)):
        augmented_image = seq(image=image)
        data_images.append(augmented_image)
        labels.append(label)

    return data_images, labels
