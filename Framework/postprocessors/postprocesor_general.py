import os
from typing import Callable
from abc import ABC, abstractmethod
import numpy as np
from Framework.postprocessors.postprocessor_functions import mean_labels_over_epochs
from Framework.utils.utils import load_txt

import matplotlib.pyplot as plt


class PostprocessorGeneral(ABC):
    def __init__(self):
        self.result_folder_path = None
        self.train_score_over_epoch_full_path = None
        self.valid_score_over_epoch_full_path = None
        self.valid_score_over_epoch_per_batch_file_name = None
        self.train_score_final_file_full_path = None

        self.measured_condition = None

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

    @abstractmethod
    def estimate_decision_lines(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass
