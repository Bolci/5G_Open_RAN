import numpy as np
from Framework.postprocessors.postprocesor_general import PostprocessorGeneral
from Framework.postprocessors.postprocessor_functions import split_score_by_labels
import matplotlib.pyplot as plt
import os
from typing import Callable

class ThresholdEstimator(PostprocessorGeneral):
    def __init__(self):
        super().__init__()

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
            no_class_0 = len(np.where(boundary_score <= data_class_0)[0])
            no_class_1 = len(np.where(boundary_score > data_class_1)[0])
            no_class_all.append((no_class_1 + no_class_0) / (no_all_class_0 + no_all_class_1))
            boundary_scores.append(boundary_score)
        return np.asarray(no_class_all), np.asarray(boundary_scores)

    def estimate_threshold_on_valid_data(self,
                                         use_epochs:int = 5,
                                         no_steps_to_estimate:int = 200):
        train_scores, valid_scores = self.load_files_final_metrics()
        classification_score_over_threshols, threshold_values = self.calculate_values_for_threshold_diagram(valid_scores,
                                                                                    no_steps_to_estimate=no_steps_to_estimate,
                                                                                    use_epochs=use_epochs)
        idx_max = np.argmax(classification_score_over_threshols)
        threshold = threshold_values[idx_max]
        classification_score = classification_score_over_threshols[idx_max]
        return threshold, classification_score, classification_score_over_threshols, threshold_values

    def get_score_on_test_data(self,
                               test_loop: Callable,
                               estimated_threshold_from_valid_data: int,
                               use_epochs: int = 5,
                               no_steps_to_estimate: int = 200,
                              ):
        classification_score_test_0, classification_score_test_1, predicted_results = (
            test_loop(estimated_threshold_from_valid_data))


        (classification_score_over_threshold_test,
         threshold_values) = self.calculate_values_for_threshold_diagram(predicted_results,
                                                                         no_steps_to_estimate,
                                                                         use_epochs)

        classification_score = np.max([classification_score_test_0, classification_score_test_1])

        return  classification_score, classification_score_over_threshold_test, threshold_values


    def calculate_classification_score(self,
                                       testing_loop,
                                       use_epochs: int = 5,
                                       no_steps_to_estimate: int = 200,
                                       prepare_fig: bool = True,
                                       save_fig: bool = True):

        (threshold_valid,
         classification_score_valid,
         classification_score_over_threshold_valid,
         threshold_values_valid) = self.estimate_threshold_on_valid_data(use_epochs=use_epochs,
                                                                         no_steps_to_estimate=no_steps_to_estimate)

        (classification_score_on_test,
         classification_score_over_threshold_test,
         threshold_values_test) = self.get_score_on_test_data(test_loop=testing_loop,
                                                              estimated_threshold_from_valid_data=threshold_valid,
                                                              use_epochs=use_epochs,
                                                              no_steps_to_estimate=no_steps_to_estimate)

        fig = None
        fig_2 = None
        if prepare_fig:
            #valid data fig
            fig, ax = plt.subplots()
            ax.vlines(threshold_valid, ymin=0, ymax=1, linestyles='dashed', alpha=0.5, color='blue',
                      label='valid_threshold')
            ax.plot(threshold_values_valid, classification_score_over_threshold_valid)
            ax.plot(threshold_valid, classification_score_valid, 'r*', label='score from valid threshold')
            ax.grid()
            ax.set_ylim([0, 1])
            ax.set_xlim([threshold_values_valid[0], threshold_values_valid[-1]])
            ax.set_title(
                f"Classification score = {classification_score_valid:.4f}, \n Threshold={threshold_valid:.8f} on validation data")
            ax.set_xlabel('Metrics score')
            ax.set_ylabel('Classification score [%]')

            # test data fig
            fig2, ax2 = plt.subplots()
            ax2.vlines(threshold_valid, ymin=0, ymax=1, linestyles='dashed', alpha=0.5)
            ax2.plot(threshold_values_test, classification_score_over_threshold_test)
            ax2.plot(threshold_valid, classification_score_on_test, 'g*')
            ax2.grid()
            ax2.set_ylim([0, 1])
            ax2.set_xlim([threshold_values_test[0], threshold_values_test[-1]])
            ax2.set_title(
                f"Classification score = {classification_score_on_test:.4f} on testing data, \n Threshold={threshold_valid:.8f} from valid data")
            ax2.set_xlabel('Metrics score')
            ax2.set_ylabel('Classification score [%]')


            if save_fig:
                saving_path = os.path.join(self.result_folder_path, 'threshols_on_testing_data.png')
                fig2.savefig(saving_path)

                saving_path = os.path.join(self.result_folder_path, 'threshols_estimation_on_validation_data.png')
                fig.savefig(saving_path)


        return classification_score_valid, classification_score_on_test, [fig, fig_2]



