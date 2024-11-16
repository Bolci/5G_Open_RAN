from Framework.postprocessors.threshold_estimator import ThresholdEstimator
from Framework.postprocessors.PDF_comparator import PDFComparator

import matplotlib.pyplot as plt
from typing import Callable
import numpy as np
import os
from copy import copy

class Tester:
    """
    A class to handle testing and estimation of decision lines using different postprocessors.
    """

    def __init__(self,
                 result_folder_path: str,
                 attempt_name: str,
                 train_score_over_epoch_file_name: str,
                 valid_score_over_epoch_file_name: str,
                 valid_score_over_epoch_per_batch_file_name: str,
                 train_score_final_file_name: str):
        """
        Initializes the Tester with paths and filenames for score files.

        :param result_folder_path: Path to the result folder.
        :param attempt_name: Name of the attempt.
        :param train_score_over_epoch_file_name: File name for training scores over epochs.
        :param valid_score_over_epoch_file_name: File name for validation scores over epochs.
        :param valid_score_over_epoch_per_batch_file_name: File name for validation scores per batch.
        :param train_score_final_file_name: File name for final training scores.
        """
        self.result_folder_path = result_folder_path
        self.attempt_name = attempt_name
        self.train_score_over_epoch_file_name = train_score_over_epoch_file_name
        self.valid_score_over_epoch_file_name = valid_score_over_epoch_file_name
        self.valid_score_over_epoch_per_batch_file_name = valid_score_over_epoch_per_batch_file_name
        self.train_score_final_file_name = train_score_final_file_name

        threshold_estimator = ThresholdEstimator()
        threshold_estimator.set_paths(result_folder_path=result_folder_path,
                                      attempt_name=attempt_name,
                                      train_score_over_epoch_file_name=train_score_over_epoch_file_name,
                                      valid_score_over_epoch_file_name=valid_score_over_epoch_file_name,
                                      valid_score_over_epoch_per_batch_file_name=valid_score_over_epoch_per_batch_file_name,
                                      train_score_final_file_name=train_score_final_file_name)

        pdf_comparator = PDFComparator()
        pdf_comparator.set_paths(result_folder_path=result_folder_path,
                                 attempt_name=attempt_name,
                                 train_score_over_epoch_file_name=train_score_over_epoch_file_name,
                                 valid_score_over_epoch_file_name=valid_score_over_epoch_file_name,
                                 valid_score_over_epoch_per_batch_file_name=valid_score_over_epoch_per_batch_file_name,
                                 train_score_final_file_name=train_score_final_file_name)

        self.tester_buffer = {'threshold_estimator': threshold_estimator,
                              'pdf_comparator': pdf_comparator}

    def estimate_decision_lines(self,
                                use_epochs: int = 1,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = True,
                                save_figs: bool = True,
                                figs_label: str = "valid_scores_over_threshold"):
        """
        Estimates decision lines using the postprocessors in the tester buffer.

        :param use_epochs: Number of epochs to use for estimation.
        :param no_steps_to_estimate: Number of steps to estimate.
        :param prepare_figs: Whether to prepare figures.
        :param save_figs: Whether to save figures.
        :param figs_label: Label for the figures.
        :return: A dictionary containing validation scores.
        """
        valid_scores = {}
        for single_tester_name, single_tester in self.tester_buffer.items():
            threshold, classification_score = single_tester.estimate_decision_lines(use_epochs=use_epochs,
                                                                                    no_steps_to_estimate=no_steps_to_estimate,
                                                                                    save_figs=save_figs,
                                                                                    prepare_figs=prepare_figs,
                                                                                    figs_label=f'{figs_label}_{single_tester_name}')
            valid_scores[single_tester_name] = copy(classification_score)

        return {'Validation scores': valid_scores}

    def test_data(self,
                  testing_loop: Callable,
                  use_epochs: int = 1,
                  no_steps_to_estimate: int = 200,
                  prepare_figs: bool = True,
                  save_figs: bool = True,
                  figs_label: str = "test_scores_over_threshold"):
        """
        Tests the data using the postprocessors in the tester buffer.

        :param testing_loop: A callable representing the testing loop.
        :param use_epochs: Number of epochs to use for testing.
        :param no_steps_to_estimate: Number of steps to estimate.
        :param prepare_figs: Whether to prepare figures.
        :param save_figs: Whether to save figures.
        :param figs_label: Label for the figures.
        :return: A dictionary containing testing scores.
        """
        testing_scores = {}

        for single_tester_name, single_tester in self.tester_buffer.items():
            classification_score_on_test = single_tester.test(testing_loop,
                                                              use_epochs=use_epochs,
                                                              no_steps_to_estimate=no_steps_to_estimate,
                                                              prepare_figs=prepare_figs,
                                                              save_figs=save_figs,
                                                              figs_label=figs_label)
            testing_scores[single_tester_name] = copy(classification_score_on_test)

        return {'Testing scores': testing_scores}