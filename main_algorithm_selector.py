from argparse import ArgumentParser

from CNNmodels import ResNet, VGG, UNet
from ViTmodels import ViTL16
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
        train_data, val_data = image_load.load_images()
        VGG.start_procedure(train_data=train_data, validation_data=val_data)

    if args.algorithm == "ResNet":
        train_data, val_data = image_load.load_images()
        ResNet.start_procedure(train_data=train_data, validation_data=val_data)
    if args.algorithm == "UNet":
        data_set, labels = image_load.load_images_method_2()

        joined_lists = list(zip(data_set, labels))
        np.random.shuffle(joined_lists)  # Shuffle "joined_lists" in place
        data_set, labels = zip(*joined_lists)  # Undo joining

        UNet.start_procedure(train_data=data_set, labels=labels)

    if args.algorithm == "ViT-L16":
        data_set, labels = image_load.load_images_method_2()

        joined_lists = list(zip(data_set, labels))
        np.random.shuffle(joined_lists)  # Shuffle "joined_lists" in place
        data_set, labels = zip(*joined_lists)  # Undo joining


        model = ViTL16.start_procedure(data_set, labels, transformer_layers=16)


