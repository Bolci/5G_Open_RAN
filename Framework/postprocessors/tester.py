from Framework.postprocessors.threshold_estimator import ThresholdEstimator
from Framework.postprocessors.PDF_comparator import PDFComparator

import matplotlib.pyplot as plt
from typing import Callable
import numpy as np
import os
from copy import copy

class Tester:
    def __init__(self,
                 result_folder_path: str,
                 attempt_name: str,
                 train_score_over_epoch_file_name: str,
                 valid_score_over_epoch_file_name: str,
                 valid_score_over_epoch_per_batch_file_name: str,
                 train_score_final_file_name: str):
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
                                use_epochs: int = 5,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = True,
                                save_figs: bool = True
                                ):
        valid_scores = {}
        for single_tester_name, single_tester in self.tester_buffer.items():
            threshold, classification_score = single_tester.estimate_decision_lines()
            valid_scores[single_tester_name] = copy(classification_score)

        return {'Validation scores': valid_scores}

    def test_data(self,
                  testing_loop: Callable,
                  use_epochs: int = 5,
                  no_steps_to_estimate: int = 200,
                  prepare_figs: bool = True,
                  save_figs: bool = True):
        
        testing_scores = {}
        all_figs = {}
        for single_tester_name, single_tester in self.tester_buffer.items():
            classification_score_on_test = single_tester.test(testing_loop)
            testing_scores[single_tester_name] = copy(classification_score_on_test)

        return {'Testing scores': testing_scores}
