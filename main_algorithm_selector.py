from argparse import ArgumentParser

from CNNmodels import ResNet, VGG, UNet
import ViTmodels

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
        train_data, val_data = image_load.load_images(width=256,height=256)
        UNet.start_procedure(train_data=train_data, validation_data=val_data)




