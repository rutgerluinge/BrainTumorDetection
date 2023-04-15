from argparse import ArgumentParser

from CNNmodels import ResNet, VGG, UNet
from ViTmodels import ViTL16, vit16l, ViTB16
import numpy as np


import image_load

if __name__ == '__main__':
    parser = ArgumentParser(prog='CNN/VIT algorithm selector',
                            description='selects a given algorithm and performs it on the dataset',
                            epilog='The following algorithms can be used as argument:\n'
                                   'CNN:| VGG, ResNet, U-Net, Inception | ViT: | B16, L16, H14 |')

    parser.add_argument("algorithm")
    args = parser.parse_args()

    if args.algorithm == "VGG":
        data_set, labels = image_load.load_images_method_2(224)
        data_set, labels = image_load.shuffle_data(data_set, labels)
        VGG.start_procedure(data=data_set, labels=labels)

    if args.algorithm == "ResNet":
        train_data, val_data = image_load.load_images()
        ResNet.start_procedure(train_data=train_data, validation_data=val_data)

    if args.algorithm == "UNet":
        data_set, labels = image_load.load_images_method_2(256)
        data_set, labels = image_load.shuffle_data(data_set, labels)

        UNet.start_procedure(data=data_set, labels=labels)

    if args.algorithm == "B16":
        data_set, labels = image_load.load_images_method_2(224)
        data_set, labels = image_load.shuffle_data(data_set, labels)

        model = ViTB16.start_procedure(data=data_set, labels=labels, size=224)

    if args.algorithm == "B32":
        data_set, labels = image_load.load_images_method_2(224)
        data_set, labels = image_load.shuffle_data(data_set, labels)

        model = ViTB16.start_procedure(data=data_set, labels=labels, size=224)    

    if args.algorithm == "test":
        data_set, labels = image_load.load_images_method_2(256)
        data_set, labels = image_load.shuffle_data(data_set, labels)
        print(f"len data: {len(data_set)}, labels: {len(labels)}")

        augmented_images, augmented_labels = image_load.data_augmentation(data_set, labels)
        print(f"len data: {len(augmented_images)}, labels: {len(augmented_labels)}")

