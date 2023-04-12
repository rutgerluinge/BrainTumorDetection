import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Dict, List
from keras.models import Model
import numpy as np
import keras.backend as K

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple"]  # nice standard matplotlib colors


def loss_accuracy_graph(data_dict: Dict[str, Model]):
    """:param data_dict dictionary which maps name of architecture to the keras model: dict["resnet"] = Model(resnet)"""
    fig, axs = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.4)

    # compute losses (train and validation)
    color_idx = 0
    for key, value in data_dict.items():
        axs[0].plot(value.history['loss'], label=f"{key}_train", color=colors[color_idx])
        axs[0].plot(value.history['val_loss'], label=f"{key}_val", linestyle="--", color=colors[color_idx])
        color_idx += 1

    axs[0].set_title("Binary cross-entropy loss")
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    # compute accuracy (train and validation)

    color_idx = 0
    for key, value in data_dict.items():
        axs[1].plot(value.history['binary_accuracy'], label=f"{key}_train", color=colors[color_idx])
        axs[1].plot(value.history['val_binary_accuracy'], label=f"{key}_val", linestyle="--", color=colors[color_idx])
        color_idx += 1

    axs[1].set_title("Binary Accuracy")
    axs[1].legend()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')


def bar_accuracy_plot(result_dict: Dict[str, List[float, float]]):
    """can be used to showcase all algorithms accuracy from test data"""
    """:param should be a dictionary, where the key is the name of the algorithm. The value of the dictionary,"""
    """ is the outcome of the corresponding model.evaluate outcome. so: dict["resnet"] = resnet.evaluate(x_test, y_test)"""
    fig, ax = plt.subplots()

    losses = [loss_acc[0] for loss_acc in result_dict.values()]  # can be used if we want
    bin_accuracies = [loss_acc[1] for loss_acc in result_dict.values()]

    ax.bar(result_dict.keys(), bin_accuracies, color="tab:cyan")

    # write accuracy within bar (looks nice :)
    for i, v in enumerate(bin_accuracies):
        ax.text(i, v / 2, str(round(v, 3)), color='white', fontweight='bold', ha='center', va='center')

    ax.set_ylim(0, 1)
    ax.set_title("Accuracy of All Models (test data)")
    ax.set_ylabel("Binary Accuracy")


def scatter_plot(model_dict: Dict[str, Model], result_dict: Dict[str, List[float, float]]):
    """:param model_dict maps the name of the model to the actual keras model (Model)
       :param result_dict maps name of model (correspond with model_names) to result (same as function above):
        dict["resnet"] = resnet.evaluate(x_test, y_test)"""

    # scatterplot accuracy of the model vs amount of parameters

    def millions_formatter(x, pos):
        return '{:.1f}M'.format(x * 1e-6)

    model_param_dict = {}

    for name, model in model_dict.items():
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

        model_param_dict[name] = [non_trainable_count, trainable_count]

    bin_accuracies = [loss_acc[1] for loss_acc in result_dict.values()]

    non_trainable_params = [params[0] for params in model_param_dict.values()]  # 1 for trainable, 0 for non trainable
    trainable_params = [params[1] for params in model_param_dict.values()]  # 1 for trainable, 0 for non trainable
    trainable_param_size = [params / 1000 for params in trainable_params]
    total_params = np.add(trainable_params, non_trainable_params)

    plt.scatter(total_params, bin_accuracies, s=trainable_param_size, c=colors[:len(model_dict.keys())])

    for idx, name in enumerate(model_dict.keys()):
        plt.text(total_params[idx], bin_accuracies[idx], f'{name}', color="black", ha='center', va='center')

    # boundaries
    plt.xlim([min(total_params) - 150000, max(total_params) + 150000])
    plt.ylim([min(bin_accuracies) - 0.01, max(bin_accuracies) + 0.01])

    # Set the plot title and axis labels
    plt.title('Binaray Accuracy vs Model Parameters')

    # format to millions on x-axis
    formatter = ticker.FuncFormatter(millions_formatter)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.xlabel('Total Parameters')
    plt.ylabel('Binary Accuracy')

    # Show the plot
    plt.show()
