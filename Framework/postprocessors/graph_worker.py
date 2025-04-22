import matplotlib.pyplot as plt
import numpy as np
from copy import copy

def get_single_dist(single_scores):
    single_scores = np.asarray(single_scores)
    indices_1 = np.where(single_scores[:, 0] == 1.)
    indices_0 = np.where(single_scores[:, 0] == 0.)


    class_1 = single_scores[indices_1][:, 1]
    class_0 = single_scores[indices_0][:, 1]

    return class_1, class_0

def get_bins_and_counts(prediction_pred_class, min_val, max_val, no_bins=20):
    prediction_pred_class = np.asarray(prediction_pred_class)
    bin_width = (max_val - min_val)/no_bins
    counts = []
    bins = []

    for x in range(no_bins):
        min_threshold = min_val + bin_width*x
        max_threshold = min_val + bin_width*(x+1)

        j = np.where((prediction_pred_class >= min_threshold) & (prediction_pred_class < max_threshold))
        counts.append(len(j[0]))
        bins.append(min_threshold)
    bins.append(max_threshold)

    counts = np.asarray(counts)
    no_all = len(prediction_pred_class)
    counts_nornalized = counts/no_all

    return bin_width, counts_nornalized, bins


def get_norm_scores_per_dataset(predctions, min_val = None, max_val = None, bins = 40):
    class_1_score, class_0_score = get_single_dist(predctions)

    if max_val is None:
        max_val = np.max(np.concatenate((class_1_score, class_0_score), axis=0))
    if min_val is None:
        min_val = np.min(np.concatenate((class_1_score, class_0_score), axis=0))

    class_0_parameters = get_bins_and_counts(class_0_score, min_val, max_val, no_bins=bins)

    if not len(class_1_score) == 0:
        class_1_parameters = get_bins_and_counts(class_1_score, min_val, max_val, no_bins=bins)
    else:
        class_1_parameters = ([],[],[])

    return class_0_parameters, class_1_parameters

def get_global_min_max(valid_predictions, test_predictions):
    class_0_score, class_1_score = get_single_dist(valid_predictions)

    if not len(class_1_score) == 0:
        valid_min = np.min(np.concatenate((class_1_score, class_0_score), axis=0))
        valid_max = np.max(np.concatenate((class_1_score, class_0_score), axis=0))
    else:
        valid_min = np.min(class_0_score)
        valid_max = np.max(class_0_score)

    mins = [valid_min]
    maxs = [valid_max]

    tst = list(test_predictions[0].keys())[0]
    for dataset_score in test_predictions:
        class_0_parameters, class_1_parameters = get_single_dist(dataset_score[tst])

        tst_min = np.min(np.concatenate((class_1_parameters, class_0_parameters), axis=0))
        tst_max = np.max(np.concatenate((class_1_parameters, class_0_parameters), axis=0))

        mins.append(tst_min)
        maxs.append(tst_max)

    _max = np.max(np.asarray(maxs))
    _min = np.min(np.asarray(mins))

    return _min, _max


def get_distribution_plot(valid_predictions, test_predictions, performance, metrics_buffer, decision_lines, plot_names):
    get_global_min_max(valid_predictions, test_predictions)
    no_datasets_test = len(test_predictions)

    min_bin, max_bin = get_global_min_max(valid_predictions, test_predictions)
    tst = list(test_predictions[0].keys())[0]
    class_0_parameters, class_1_parameters = get_norm_scores_per_dataset(valid_predictions, min_bin, max_bin)

    fig, ax = plt.subplots(2,no_datasets_test+1, figsize=(5*(no_datasets_test+1), 10))
    ax[0, 0].bar(class_0_parameters[2][:-1], class_0_parameters[1], width=class_0_parameters[0], alpha=0.5, color='r',
              label='Normalized Histogram valid class 0')
    ax[0, 0].bar(class_1_parameters[2][:-1], class_1_parameters[1], width=class_1_parameters[0], alpha=0.5, color='b',
              label='Normalized Histogram valid class 1')


    # add decision line
    if decision_lines is not None:
        for tester_name, bounds in decision_lines:
            if "min_max" in tester_name:
                label = "Min-Max"
                color = "g"
            elif "std" in tester_name:
                label = "STD"
                color = "y"
            elif "mad" in tester_name:
                label = "MAD"
                color = "m"
            else:
                label = "Unknown"
                color = "k"
            if bounds[0] is not None and bounds[1] is not None:
                ax[0, 0].axvline(x=bounds[0], color=color, linestyle='--', label=f"{label}-lower bound")
                ax[0, 0].axvline(x=bounds[1], color=color, linestyle='--', label=f"{label}-upper bound")
    ax[0, 0].set_title('Validation Error distribution')
    ax[0, 0].set_xlabel('Anomaly Score')
    ax[0, 0].set_ylabel('Density')
    ax[0, 0].legend()
    ax[0, 0].set_ylim([0,0.5])

    ax[0, 0].grid(True)


    subfigs = []
    for id_dataset, dataset_score in enumerate(test_predictions):
        class_0_parameters, class_1_parameters = get_norm_scores_per_dataset(dataset_score[tst], min_bin, max_bin)
        ax[0, id_dataset+1].bar(class_0_parameters[2][:-1], class_0_parameters[1], width=class_0_parameters[0], alpha=0.5, color='r',
              label='Normalized Histogram valid class 0')
        ax[0, id_dataset+1].bar(class_1_parameters[2][:-1], class_1_parameters[1], width=class_1_parameters[0], alpha=0.5, color='b',
              label='Normalized Histogram valid class 1')

        # add decision line
        if decision_lines is not None:
            for tester_name, bounds in decision_lines:
                if "min_max" in tester_name:
                    label = "Min-Max"
                    color = "g"
                elif "std" in tester_name:
                    label = "STD"
                    color = "y"
                elif "mad" in tester_name:
                    label = "MAD"
                    color = "m"
                else:
                    label = "Unknown"
                    color = "k"
                if bounds[0] is not None and bounds[1] is not None:
                    ax[0, id_dataset+1].axvline(x=bounds[0],color=color, linestyle='--', label=f"{label}-lower bound")
                    ax[0, id_dataset+1].axvline(x=bounds[1],color=color, linestyle='--', label=f"{label}-upper bound")


        ax[0, id_dataset+1].set_ylim([0, 0.5])
        ax[0, id_dataset+1].legend()
        ax[0, id_dataset+1].grid(True)
        ax[0, id_dataset+1].set_xlabel('Anomaly Score')
        ax[0, id_dataset+1].set_ylabel('Density')


        scores = ""
        for value in performance[id_dataset].values():
            scores += f"{value:.3f}, "
        ax[0, id_dataset+1].set_title(f'{plot_names[id_dataset]}')

        # Add table with metrics below the graph
        table_data = []
        for key, metrics in metrics_buffer[id_dataset].items():
            value = [f"{value:.3f}" for value in metrics]
            table_data.append([key] + value)
        # table_data = [[key, f"{value}"] for key, values in metrics_buffer[id_dataset].items()]
        table = ax[1, id_dataset+1].table(cellText=table_data, colLabels=["Estim.", "Prec.", "Recall", "F1"], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.5, 1.5)  # Increase the width of the first column
        for key, cell in table.get_celld().items():
            if key[1] == 0:  # First column
                cell.set_width(0.55)
            else:
                cell.set_width(0.15)
        ax[1, id_dataset+1].axis('off')
        ax[1, 0].axis('off')

        fig2, ax2 = plt.subplots(2, 1, figsize=(5 , 10))
        class_0_parameters, class_1_parameters = get_norm_scores_per_dataset(dataset_score[tst], min_bin, max_bin)
        ax2[0].bar(class_0_parameters[2][:-1], class_0_parameters[1], width=class_0_parameters[0],
                                  alpha=0.5, color='r',
                                  label='Normalized Histogram valid class 0')
        ax2[0].bar(class_1_parameters[2][:-1], class_1_parameters[1], width=class_1_parameters[0],
                                  alpha=0.5, color='b',
                                  label='Normalized Histogram valid class 1')

        # add decision line
        if decision_lines is not None:
            for tester_name, bounds in decision_lines:
                if "min_max" in tester_name:
                    label = "Min-Max"
                    color = "g"
                elif "std" in tester_name:
                    label = "STD"
                    color = "y"
                elif "mad" in tester_name:
                    label = "MAD"
                    color = "m"
                else:
                    label = "Unknown"
                    color = "k"
                if bounds[0] is not None and bounds[1] is not None:
                    ax2[0].axvline(x=bounds[0],color=color, linestyle='--', label=f"{label}-lower bound")
                    ax2[0].axvline(x=bounds[1],color=color, linestyle='--', label=f"{label}-upper bound")

        ax2[0].set_ylim([0, 0.5])
        ax2[0].legend()
        ax2[0].grid(True)
        ax2[0].set_xlabel('Anomaly Score')
        ax2[0].set_ylabel('Density')

        scores = ""
        for value in performance[id_dataset].values():
            scores += f"{value:.3f}, "
        ax2[0].set_title(f'{plot_names[id_dataset]}')

        # Add table with metrics below the graph
        table_data = []
        for key, metrics in metrics_buffer[id_dataset].items():
            value = [f"{value:.3f}" for value in metrics]
            table_data.append([key] + value)
        # table_data = [[key, f"{value}"] for key, values in metrics_buffer[id_dataset].items()]
        table = ax2[1].table(cellText=table_data, colLabels=["Estim.", "Prec.", "Recall", "F1"],
                                            loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.5, 1.5)  # Increase the width of the first column
        for key, cell in table.get_celld().items():
            if key[1] == 0:  # First column
                cell.set_width(0.55)
            else:
                cell.set_width(0.15)
        ax2[1].axis('off')
        ax2[1].axis('off')
        fig2.suptitle("Error distributions")
        fig2.tight_layout()
        plt.close(fig2)
        subfigs.append(fig2)


    fig.suptitle("Error distributions")
    fig.tight_layout()
    plt.close(fig)

    return fig, subfigs


