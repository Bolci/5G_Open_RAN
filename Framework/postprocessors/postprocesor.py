import os
from typing import Callable

import numpy as np
from Framework.postprocessors.postprocessor_functions import mean_labels_over_epochs
from Framework.utils.utils import load_txt
from Framework.postprocessors.postprocessor_functions import split_score_by_labels
import matplotlib.pyplot as plt


class Postprocessor:
    def __init__(self):
        self.result_folder_path = None
        self.train_score_over_epoch_full_path = None
        self.valid_score_over_epoch_full_path = None
        self.valid_score_over_epoch_per_batch_file_name = None
        self.train_score_final_file_full_path = None

    def set_paths(self,
                  result_folder_path: str,
                  attempt_name: str,
                  train_score_over_epoch_file_name: str,
                  valid_score_over_epoch_file_name: str,
                  valid_score_over_epoch_per_batch_file_name: str,
                  train_score_final_file_name: str):
        attempt_folder_name = os.path.join(result_folder_path, attempt_name)

        self.result_folder_path = attempt_folder_name
        self.train_score_over_epoch_full_path = os.path.join(attempt_folder_name, train_score_over_epoch_file_name)
        self.valid_score_over_epoch_full_path = os.path.join(attempt_folder_name, valid_score_over_epoch_file_name)
        self.valid_score_over_epoch_per_batch_file_name = os.path.join(attempt_folder_name, valid_score_over_epoch_per_batch_file_name)
        self.train_score_final_file_full_path = os.path.join(attempt_folder_name, train_score_final_file_name)

    def load_files_final_metrics(self):
        train_scores = load_txt(self.train_score_final_file_full_path)
        valid_scores = load_txt(self.valid_score_over_epoch_per_batch_file_name)[-1]
        return train_scores, valid_scores


    def load_and_parse_valid_per_batch_per_epoch(self):
        valid_scores = load_txt(self.valid_score_over_epoch_per_batch_file_name)
        valid_scores = mean_labels_over_epochs(valid_scores)
        return valid_scores['Class_0'], valid_scores['Class_1']


    def load_files_over_epochs(self):
        train_scores = load_txt(self.train_score_over_epoch_full_path)
        valid_scores = load_txt(self.valid_score_over_epoch_full_path)
        return train_scores, valid_scores


    def get_threshold_limits(self, use_epochs:int = 5):
        train_over_epoch, _ = self.load_files_over_epochs()
        valid_over_epoch_class_0, valid_over_epoch_class_1 = self.load_and_parse_valid_per_batch_per_epoch()

        train_scores = np.mean(np.asarray(train_over_epoch[-use_epochs:]))
        valid_scores_0 = np.mean(np.asarray(valid_over_epoch_class_0[-use_epochs:]))
        valid_scores_1 = np.mean(np.asarray(valid_over_epoch_class_1[-use_epochs:]))
        all_scores = np.asarray([train_scores, valid_scores_0, valid_scores_1])
        max_score = np.max(all_scores)
        min_score = np.min(all_scores)

        max_score += 0.5 * max_score
        min_score -= 0.5 * min_score

        return min_score, max_score


    def calculate_values_for_threshold_diagram(self, data, no_steps_to_estimate, use_epochs):
        min_score, max_score = self.get_threshold_limits(use_epochs=use_epochs)
        ds = (max_score - min_score) / no_steps_to_estimate

        boundary_scores = []
        no_class_all = []

        data_class_1, data_class_0 = split_score_by_labels(data)
        no_all_class_0 = len(data_class_0)
        no_all_class_1 = len(data_class_1)

        for x in range(200):
            boundary_score = min_score + ds * x
            no_class_0 = len(np.where(boundary_score <= data_class_0)[0])
            no_class_1 = len(np.where(boundary_score > data_class_1)[0])
            no_class_all.append((no_class_1 + no_class_0) / (no_all_class_0 + no_all_class_1))
            boundary_scores.append(boundary_score)
        return np.asarray(no_class_all), np.asarray(boundary_scores)


    def estimate_threshold_on_valid_data(self, use_epochs:int = 5,
                                         no_steps_to_estimate:int = 200,
                                         prepare_fig:bool = True,
                                         save_fig: bool = True):
        train_scores, valid_scores = self.load_files_final_metrics()
        no_class_all, boundary_scores = self.calculate_values_for_threshold_diagram(valid_scores,
                                                                                    no_steps_to_estimate=no_steps_to_estimate,
                                                                                    use_epochs=use_epochs)
        boundary_scores = np.asarray(boundary_scores)
        no_class_all = np.asarray(no_class_all)

        idx_max = np.argmax(no_class_all)
        threshold = boundary_scores[idx_max]
        classification_score = no_class_all[idx_max]

        fig = None
        if prepare_fig:
            fig, ax  = plt.subplots()
            ax.vlines(boundary_scores[idx_max], ymin=0, ymax=1, linestyles='dashed', alpha=0.5)
            ax.plot(boundary_scores, no_class_all)
            ax.plot(threshold,classification_score , 'r*')
            ax.grid()
            ax.set_ylim([0,1])
            ax.set_xlim([boundary_scores[0], boundary_scores[-1]])
            ax.set_title(f"Classification score = {classification_score:.4f}, \n Threshold={threshold:.8f} on validation data")
            ax.set_xlabel('Metrics score')
            ax.set_ylabel('Classification score [%]')

            if save_fig:
                saving_path = os.path.join(self.result_folder_path, 'threshols_estimation_on_validation_data.png')
                fig.savefig(saving_path)

        return threshold, classification_score, fig

    def test_data(self,
                  testing_threshold_from_valid: float,
                  testing_loop: Callable,
                  use_epochs:int = 5,
                  no_steps_to_estimate:int = 200,
                  prepare_fig:bool = True,
                  save_fig: bool = True):
        classification_score_test_0, classification_score_test_1, predicted_results = testing_loop()
        no_class_all, boundary_scores = self.calculate_values_for_threshold_diagram(predicted_results, no_steps_to_estimate, use_epochs)
        boundary_scores = np.asarray(boundary_scores)
        no_class_all = np.asarray(no_class_all)
        max_valid_score = np.max(np.asarray([classification_score_test_0, classification_score_test_1]))

        fig = None
        if prepare_fig:
            fig, ax = plt.subplots()
            ax.vlines(testing_threshold_from_valid, ymin=0, ymax=1, linestyles='dashed', alpha=0.5, color = 'green', label='valid_threshold')
            ax.plot(boundary_scores, no_class_all)
            ax.plot(testing_threshold_from_valid, max_valid_score, 'g*', label = 'score from valid threshold')
            ax.grid()
            ax.set_ylim([0, 1])
            ax.set_xlim([boundary_scores[0], boundary_scores[-1]])
            ax.set_title(
                f"Classification score = {max_valid_score:.4f} on testing data, \n Threshold={testing_threshold_from_valid:.8f} from valid data")
            ax.set_xlabel('Metrics score')
            ax.set_ylabel('Classification score [%]')

            if save_fig:
                saving_path = os.path.join(self.result_folder_path, 'threshols_on_testing_data.png')
                fig.savefig(saving_path)

        return classification_score_test_0, classification_score_test_1, predicted_results, fig



