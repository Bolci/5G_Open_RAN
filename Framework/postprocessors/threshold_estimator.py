import numpy as np
from Framework.postprocessors.postprocesor_general import PostprocessorGeneral
from Framework.postprocessors.postprocessor_functions import split_score_by_labels
import matplotlib.pyplot as plt
import os
from typing import Callable


class ThresholdEstimator(PostprocessorGeneral):
    def __init__(self):
        super().__init__()

        self.classification_score_over_thresholds_valid = None
        self.threshold_values = None

    def get_threshold_limits(self, use_epochs:int = 1):
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

    def calculate_values_for_threshold_diagram(self, data: int, no_steps_to_estimate: int, use_epochs: int):
        min_score, max_score = self.get_threshold_limits(use_epochs=use_epochs)
        ds = (max_score - min_score) / no_steps_to_estimate

        boundary_scores = []
        no_class_all = []

        data_class_1, data_class_0 = split_score_by_labels(data)
        no_all_class_0 = len(data_class_0)
        no_all_class_1 = len(data_class_1)

        for x in range(200):
            boundary_score = min_score + ds * x
            no_class_0 = len(np.where(boundary_score >= data_class_0[:, 1])[0])
            no_class_1 = len(np.where(boundary_score < data_class_1[:, 1])[0])

            no_class_all.append((no_class_1 + no_class_0) / (no_all_class_0 + no_all_class_1))

            boundary_scores.append(boundary_score)

        all_class_return = no_class_all

        return np.asarray(all_class_return), np.asarray(boundary_scores)

    @staticmethod
    def estimate_threshold(classification_score_over_thresholds, threshold_values):
        idx_max = np.argmax(classification_score_over_thresholds)
        threshold = threshold_values[idx_max]
        classification_score = classification_score_over_thresholds[idx_max]
        return threshold, classification_score

    def get_score_on_test_data(self,
                               test_loop: Callable,
                               use_epochs: int,
                               no_steps_to_estimate: int,
                              ):

        if self.measured_decision_line is None:
            raise Exception('Threshold has not been estimated yet')

        classification_score_test, predicted_results = (test_loop(self.measured_decision_line))

        (classification_score_over_threshold_test,
         threshold_values) = self.calculate_values_for_threshold_diagram(predicted_results,
                                                                         no_steps_to_estimate,
                                                                         use_epochs)

        return classification_score_test, classification_score_over_threshold_test, threshold_values


    def estimate_decision_lines(self,
                                use_epochs:int = 5,
                                no_steps_to_estimate:int = 200,
                                prepare_figs: bool = False,
                                save_figs: bool = False,
                                figs_label: str = "",
                                ):
        train_scores, valid_scores = self.load_files_final_metrics()

        classification_score_over_thresholds, threshold_values = self.calculate_values_for_threshold_diagram(
                                                                                valid_scores,
                                                                                no_steps_to_estimate=no_steps_to_estimate,
                                                                                use_epochs=use_epochs)

        threshold, classification_score = self.estimate_threshold(classification_score_over_thresholds, threshold_values)

        self.classification_score_over_thresholds_valid = classification_score_over_thresholds
        self.threshold_values = threshold_values
        self.measured_decision_line = threshold

        if prepare_figs:
            self.get_fig(
                results_scores=classification_score_over_thresholds,
                score_in_threshold=classification_score,
                saving_label=figs_label,
                save_fig=save_figs)

        return threshold, classification_score

    def test(self,
             testing_loop,
             use_epochs: int = 5,
             no_steps_to_estimate: int = 200,
             prepare_figs: bool = False,
             save_figs: bool = False,
             figs_label: str = "",
             ):

        (classification_score_on_test,
         classification_score_over_threshold_test,
         threshold_values_test) = self.get_score_on_test_data(test_loop=testing_loop,
                                                              use_epochs=use_epochs,
                                                              no_steps_to_estimate=no_steps_to_estimate)

        if prepare_figs:
            self.get_fig(
                results_scores=classification_score_over_threshold_test,
                score_in_threshold=classification_score_on_test,
                saving_label=figs_label,
                save_fig=save_figs)

        return classification_score_on_test

    def get_fig(self,
                results_scores,
                score_in_threshold,
                saving_label: str = 'fig',
                save_fig: bool = True):

        fig, ax = plt.subplots()
        ax.vlines(self.measured_decision_line, ymin=0, ymax=1, linestyles='dashed', alpha=0.5, color='blue',
                  label='valid_threshold', linewidth=2.0)
        ax.plot(self.threshold_values, results_scores, linewidth=2.0)
        ax.plot(self.measured_decision_line, score_in_threshold, 'r*', label='score from valid threshold',
                linewidth=2.5)
        ax.grid()
        ax.set_ylim([0, 1])
        ax.set_xlim([self.threshold_values[0], self.threshold_values[-1]])
        ax.set_title(
            f"Classification score = {score_in_threshold:.4f}, \n Threshold={self.measured_decision_line:.8f} on validation data")
        ax.set_xlabel('Metrics score')
        ax.set_ylabel('Classification score [%]')

        if save_fig:
            saving_path = os.path.join(self.result_folder_path, f'{saving_label}.png')
            fig.savefig(saving_path)