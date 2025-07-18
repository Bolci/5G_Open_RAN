import os
from typing import Callable
from abc import ABC, abstractmethod
import numpy as np
from Framework.postprocessors.postprocessor_functions import mean_labels_over_epochs
from Framework.utils.utils import load_txt

import matplotlib.pyplot as plt


class PostprocessorGeneral(ABC):
    """
    Abstract base class for general postprocessing tasks.
    """

    def __init__(self):
        """
        Initializes the PostprocessorGeneral with default paths and measured decision line.
        """
        self.result_folder_path = None
        self.train_score_over_epoch_full_path = None
        self.valid_score_over_epoch_full_path = None
        self.valid_score_over_epoch_per_batch_file_name = None
        self.train_score_final_file_full_path = None

        self.measured_decision_line = None
        self.sigma_estimated = False

    def set_paths(self,
                  result_folder_path: str,
                  attempt_name: str,
                  train_score_over_epoch_file_name: str,
                  valid_score_over_epoch_file_name: str,
                  valid_score_over_epoch_per_batch_file_name: str,
                  train_score_final_file_name: str):
        """
        Sets the file paths for various score files based on the provided parameters.

        :param result_folder_path: Path to the result folder.
        :param attempt_name: Name of the attempt.
        :param train_score_over_epoch_file_name: File name for training scores over epochs.
        :param valid_score_over_epoch_file_name: File name for validation scores over epochs.
        :param valid_score_over_epoch_per_batch_file_name: File name for validation scores per batch.
        :param train_score_final_file_name: File name for final training scores.
        """
        attempt_folder_name = os.path.join(result_folder_path, attempt_name)

        self.result_folder_path = attempt_folder_name
        self.train_score_over_epoch_full_path = os.path.join(attempt_folder_name, train_score_over_epoch_file_name)
        self.valid_score_over_epoch_full_path = os.path.join(attempt_folder_name, valid_score_over_epoch_file_name)
        self.valid_score_over_epoch_per_batch_file_name = os.path.join(attempt_folder_name, valid_score_over_epoch_per_batch_file_name)
        self.train_score_final_file_full_path = os.path.join(attempt_folder_name, train_score_final_file_name)

    def load_files_final_metrics(self):
        """
        Loads the final metrics from the respective files.

        :return: A tuple containing training scores and the last validation score.
        """
        # train_scores = load_txt(self.train_score_final_file_full_path)
        # valid_scores = load_txt(self.valid_score_over_epoch_per_batch_file_name)[-1]
        # switch bettween filepath and nparray
        if isinstance(self.train_score_final_file_full_path, np.ndarray):
            train_scores = self.train_score_final_file_full_path
            valid_scores = self.valid_score_over_epoch_per_batch_file_name[-1]
        else:
            train_scores = np.load(self.train_score_final_file_full_path)
            valid_scores = np.load(self.valid_score_over_epoch_per_batch_file_name)[-1]

        return train_scores, valid_scores

    def load_and_parse_valid_per_batch_per_epoch(self):
        """
        Loads and parses validation scores per batch per epoch.

        :return: A tuple containing validation scores for Class_0 and Class_1.
        """
        valid_scores = load_txt(self.valid_score_over_epoch_per_batch_file_name)
        valid_scores = mean_labels_over_epochs(valid_scores)
        return valid_scores['Class_0'], valid_scores['Class_1']

    def load_files_over_epochs(self):
        """
        Loads the scores over epochs from the respective files.

        :return: A tuple containing training scores and validation scores over epochs.
        """

        # switch bettween filepath and np.array
        if isinstance(self.train_score_over_epoch_full_path, np.ndarray):
            train_scores = self.train_score_over_epoch_full_path
            valid_scores = self.valid_score_over_epoch_full_path
        else:
            train_scores = load_txt(self.train_score_over_epoch_full_path)
            valid_scores = load_txt(self.valid_score_over_epoch_full_path)
        return train_scores, valid_scores

    @abstractmethod
    def estimate_decision_lines(self, *args, **kwargs):
        """
        Abstract method to estimate decision lines. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        """
        Abstract method to test the postprocessor. Must be implemented by subclasses.
        """
        pass