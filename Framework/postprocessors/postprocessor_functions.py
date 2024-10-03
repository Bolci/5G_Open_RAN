import numpy as np
import matplotlib.pyplot as plt


def get_epochs_axis_x(id_epoch, num,  offset):
    axis = np.asarray([id_epoch - offset] * num)
    return axis

def mean_labels_over_epochs(data):
    data = np.asarray(data)

    epochs = []
    class_1_metrics = []
    class_0_metrics = []

    for id_epoch, data_per_epoch in enumerate(data):
        indices_1 = np.where(data_per_epoch[:, 0] == 1)
        indices_0 = np.where(data_per_epoch[:, 0] == 0)

        class_1 = data_per_epoch[indices_1][:,1]
        class_0 = data_per_epoch[indices_0][:,1]

        class_1 = np.mean(class_1)
        class_0 = np.mean(class_0)

        epochs.append(id_epoch)
        class_1_metrics.append(class_1)
        class_0_metrics.append(class_0)

    return {'Epochs': epochs, 'Class_1': class_1_metrics, 'Class_0': class_0_metrics}



def plot_data_by_labels(data, saving_path):

    plt.figure()
    data = np.asarray(data)
    for id_epoch, data_per_epoch in enumerate(data):
        indices_1 = np.where(data_per_epoch[:, 0] == 1)
        indices_0 = np.where(data_per_epoch[:, 0] == 0)

        class_1 = data_per_epoch[indices_1]
        axis_x_1 = get_epochs_axis_x(id_epoch, len(class_1), offset=0.2)

        class_0 = data_per_epoch[indices_0]
        axis_x_0 = get_epochs_axis_x(id_epoch, len(class_0), offset=-0.2)

        plt.plot(axis_x_1, class_1[:,1], 'r')
        plt.plot(axis_x_0, class_0[:,1], 'b')

    plt.grid()
    plt.xlabel('epochs')
    plt.savefig(saving_path)
