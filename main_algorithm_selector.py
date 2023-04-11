from argparse import ArgumentParser

from CNNmodels import ResNet, VGG, UNet
from ViTmodels import ViTL16, vit16l
from image_load import *


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
        data_set, labels = image_load.load_images_method_2(256)
        data_set, labels = image_load.shuffle_data(data_set, labels)

        UNet.start_procedure(data=data_set, labels=labels)

    if args.algorithm == "ViT-L16":
        data_set, labels = load_images_method_2(240)
        data_set, labels = shuffle_data(data_set, labels)
        x, y, x_val, y_val, x_test, y_test = image_load.split_data(data_set, labels)

        model = ViTL16.start_procedure(x=x, y=y, x_val=x_val, y_val=y_val,transformer_layers=16)

    if args.algorithm == "vit16l":
        data_set, labels = image_load.load_images_method_2(256)
        data_set, labels = image_load.shuffle_data(data_set, labels)

        model = vit16l.start_procedure(data=data_set, labels=labels, size=256)



