import numpy as np
import matplotlib.pyplot as plt

def get_epochs_axis_x(id_epoch, num, offset):
    """
    Generates an array representing the x-axis values for epochs.

    :param id_epoch: The current epoch ID.
    :param num: The number of elements.
    :param offset: The offset to apply to the epoch ID.
    :return: A numpy array of x-axis values.
    """
    axis = np.asarray([id_epoch - offset] * num)
    return axis

def mean_labels_over_epochs(data):
    """
    Calculates the mean metrics for each class over epochs.

    :param data: np.ndarray
    :return: A dictionary with epochs, class 1 metrics, and class 0 metrics.
    """
    data = np.asarray(data)


    epochs = []
    class_1_metrics = []
    class_0_metrics = []

    for id_epoch, data_per_epoch in enumerate(data):
        indices_1 = np.where(data_per_epoch[:, 0] == 1)
        indices_0 = np.where(data_per_epoch[:, 0] == 0)

        class_1 = data_per_epoch[indices_1, 1]
        class_0 = data_per_epoch[indices_0, 1]

        class_1 = np.mean(class_1)
        class_0 = np.mean(class_0)

        epochs.append(id_epoch)
        class_1_metrics.append(class_1)
        class_0_metrics.append(class_0)

    return {'Epochs': epochs, 'Class_1': class_1_metrics, 'Class_0': class_0_metrics}

def split_score_by_labels(data):
    """
    Splits the data into two classes based on labels.

    :param data: A list or array of data with labels.
    :return: Two arrays, one for each class.
    """
    data = np.asarray(data)
    indices_1 = np.where(data[:, 0] == 1)
    indices_0 = np.where(data[:, 0] == 0)

    class_1 = data[indices_1]
    class_0 = data[indices_0]

    return class_1, class_0

def plot_data_by_labels(data, saving_path):
    """
    Plots the data for each class over epochs and saves the plot.

    :param data: A list or array of data with labels.
    :param saving_path: The path to save the plot.
    """
    plt.figure()
    data = np.asarray(data)
    for id_epoch, data_per_epoch in enumerate(data):
        indices_1 = np.where(data_per_epoch[:, 0] == 1)
        indices_0 = np.where(data_per_epoch[:, 0] == 0)

        class_1 = data_per_epoch[indices_1]
        axis_x_1 = get_epochs_axis_x(id_epoch, len(class_1), offset=0.2)

        class_0 = data_per_epoch[indices_0]
        axis_x_0 = get_epochs_axis_x(id_epoch, len(class_0), offset=-0.2)

        plt.plot(axis_x_1, class_1[:,1], 'r', alpha=0.1)
        plt.plot(axis_x_0, class_0[:,1], 'b', alpha=0.1)

    plt.grid()
    plt.xlabel('epochs')
    plt.savefig(saving_path)
