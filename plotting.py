import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List
from keras.models import Model
import numpy as np
import keras.backend as K
from sklearn.metrics import confusion_matrix


import cv2


colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:blue", "tab:orange", "tab:green",
          "tab:red", "tab:purple"]  # nice standard matplotlib colors


def loss_accuracy_graph_from_pickle(history_dict):
    """:param history_dict dictionary which maps name of architecture to the training history"""
    # plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    plt.figure(figsize=(22, 22))

    fig.subplots_adjust(wspace=0.2, hspace=0.8)

    # compute losses (train and validation)
    for key, value in history_dict.items():
        axs[0][0].plot(value[2], label=f"{key}")

    axs[0][0].set_title("Training Accuracy")
    axs[0][0].set_ylabel('Binary Accuracy')
    axs[0][0].set_ylim(0, 1.05)

    for key, value in history_dict.items():
        axs[0][1].plot(value[3])

    axs[0][1].set_ylim(0, 1.05)
    axs[0][1].set_title("Validation Accuracy")

    for key, value in history_dict.items():
        axs[1][0].plot(value[0])

    axs[1][0].set_title("Training Loss")
    axs[1][0].set_xlabel('Epoch')
    axs[1][0].set_ylabel('Binary cross-entropy loss')
    axs[1][0].set_ylim(0, 3)
    for key, value in history_dict.items():
        axs[1][1].plot(value[1])

    axs[1][1].set_title("Validation Loss")
    axs[1][1].set_xlabel('Epoch')
    axs[1][1].set_ylim(0, 3)

    fig.legend(bbox_to_anchor=(1.20, 0.5), loc='center right', borderaxespad=0., bbox_transform=fig.transFigure)
    fig.tight_layout()

def loss_accuracy_graph(model_dict: Dict[str, Model]):
    """:param data_dict dictionary which maps name of architecture to the keras model: dict["resnet"] = Model(resnet)"""
    fig, axs = plt.subplots(1, 2)
    plt.figure(figsize=(25, 7))
    fig.subplots_adjust(wspace=0.4)

    # compute losses (train and validation)
    color_idx = 0
    for key, value in model_dict.items():
        axs[0].plot(value.history['loss'], label=f"{key}_train", color=colors[color_idx])
        axs[0].plot(value.history['val_loss'], label=f"{key}_val", linestyle="--", color=colors[color_idx])
        color_idx += 1

    axs[0].set_title("Binary cross-entropy loss")
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    # compute accuracy (train and validation)

    color_idx = 0
    for key, value in model_dict.items():
        axs[1].plot(value.history['binary_accuracy'], label=f"{key}_train", color=colors[color_idx])
        axs[1].plot(value.history['val_binary_accuracy'], label=f"{key}_val", linestyle="--", color=colors[color_idx])
        color_idx += 1

    axs[1].set_title("Binary Accuracy")
    axs[1].legend()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')


def bar_accuracy_plot(result_dict: Dict[str, List[float]]):
    """can be used to showcase all algorithms accuracy from test data
       :param result_dict mapping from model name to final accuracy"""
    fig, ax = plt.subplots()

    losses = [loss_acc[1] for loss_acc in result_dict.values()]  # can be used if we want
    bin_accuracies = [loss_acc[0] for loss_acc in result_dict.values()]

    ax.bar(result_dict.keys(), bin_accuracies, color="tab:cyan")

    # write accuracy within bar (looks nice :)
    for i, v in enumerate(bin_accuracies):
        ax.text(i, v / 2, str(round(v, 3)), color='white', fontweight='bold', ha='center', va='center')

    ax.set_ylim(0, 1)
    ax.set_title("Accuracy of All Models")
    ax.set_ylabel("Binary Accuracy")


def scatter_plot(model_dict: Dict[str, Model], acc_loss_dict, flops: Dict[str, float], dot_scalar = 500):
    """:param model_dict maps the name of the model to the actual keras model (Model)
       :param acc_loss_dict dictionary containing accuracy and acc_loss_dict
       :param flops gflops dictionary
       :param dot scalar scalar to scale the scatter point size.
        """
    plt.figure(figsize=(8, 8))
    # scatterplot accuracy of the model vs amount of parameters

    def millions_formatter(x, pos):
        return '{:.1f}M'.format(x * 1e-6)


    model_param_dict = {}

    for name, model in model_dict.items():
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

        model_param_dict[name] = [non_trainable_count, trainable_count]

    bin_accuracies = [loss_acc[0] for loss_acc in acc_loss_dict.values()]

    non_trainable_params = [params[0] for params in model_param_dict.values()]  # 1 for trainable, 0 for non trainable
    trainable_params = [params[1] for params in model_param_dict.values()]  # 1 for trainable, 0 for non trainable
    flops = [value for value in flops.values()]

    total_params = np.add(trainable_params, non_trainable_params)

    s_norm = total_params / np.max(total_params)
    s_scaled = s_norm * dot_scalar  # Scale the s values by a factor of 500


    plt.scatter(flops, bin_accuracies, s=s_scaled, c=colors[:len(model_dict.keys())], alpha=0.5)

    for idx, name in enumerate(model_dict.keys()):
        plt.text(flops[idx], bin_accuracies[idx], f'{name}', color="black", ha='center', va='center')

    # boundaries
    plt.xlim([0, 40])
    plt.ylim(0, 1.05)


    # Set the plot title and axis labels
    plt.title('Accuracy vs GFLOPs vs Total Parameters')

    # format to millions on x-axis
    # formatter = ticker.FuncFormatter(millions_formatter)
    # plt.gca().xaxis.set_major_formatter(formatter)

    plt.xlabel('Operations in giga FLOPs')
    plt.ylabel('Binary Accuracy')

    # Show the plot
    plt.show()


def confusion_matrices(model_dict: Dict[str, Model], data_dict):
    """Confusion matrices for all models.
        @:param model_dict: dictionary containing all models
        @:param data_dict, containing the validation data of the model."""
    plt.figure(figsize=(20, 35))
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    x, y = 0, 0
    im = None
    for name, model in model_dict.items():
        data = data_dict[name]
        x_test = data[0]
        y_test = data[1]

        print(f"model: {name}")
        y_pred = model.predict(x_test)
        y_pred2 = []
        for item in y_pred:
            y_pred2.append(np.argmax(item))
        cm = confusion_matrix(y_test, y_pred2)

        im = plot_confusion_matrix(cm, ['yes', 'no'], axis=axs[y, x], title=f"{name}")

        x += 1
        if x % 2 == 0:
            x %= 2
            y += 1

    axs[2, 1].set_visible(False)
    fig.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)


def plot_confusion_matrix(cm, classes, axis, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('int') #/ cm.sum(axis=1)[:, np.newaxis]

    im = axis.imshow(cm, interpolation='nearest', cmap=cmap)
    axis.set_title(title)

    tick_marks = np.arange(len(classes))

    axis.set_xticks(tick_marks)
    axis.set_xticklabels(classes)

    axis.set_yticks(tick_marks)
    axis.set_yticklabels(classes)

    fmt = '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axis.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    axis.set_ylabel('True label')
    axis.set_xlabel('Predicted label')
    return im