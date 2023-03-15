from argparse import ArgumentParser

from CNNmodels import ResNet, VGG
import ViTmodels

import image_load

if __name__ == '__main__':
    parser = ArgumentParser(prog='CNN/VIT algorithm selector',
                            description='selects a given algorithm and performs it on the dataset',
                            epilog='The following algorithms can be used as argument:\n'
                                   'CNN:| VGG, ResNet, U-Net, Inception | ViT: | B16, L16, H14 |')

    parser.add_argument("algorithm")
    args = parser.parse_args()

    train_data, val_data = image_load.load_images()

    if args.algorithm == "VGG":
        VGG.start_procedure(train_data=train_data, validation_data=val_data)

    if args.algorithm == "ResNet":
        ResNet.start_procedure(train_data=train_data, validation_data=val_data)



