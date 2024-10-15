import os

from sqlalchemy import table
from Framework.postprocessors.postprocessor_functions import mean_labels_over_epochs
from Framework.utils.utils import load_txt
from Framework.postprocessors.postprocessor_functions import split_score_by_labels
import matplotlib.pyplot as plt


class Postprocessor:
    def __init__(self):
        self.result_folder_path = None
        self.train_score_file_full_path = None
        self.valid_score_file_full_path = None
        self.train_score_final_file_full_path = None

    def set_paths(self,
                  result_folder_path: str,
                  attempt_name: str,
                  train_score_paths_file_name: str,
                  valid_score_path_file_name: str,
                  train_score_final_file_name: str):
        attempt_folder_name = os.path.join(result_folder_path, attempt_name)

        self.result_folder_path = attempt_folder_name
        self.train_score_file_full_path = os.path.join(attempt_folder_name, train_score_paths_file_name)
        self.valid_score_file_full_path = os.path.join(attempt_folder_name, valid_score_path_file_name)
        self.train_score_final_file_full_path = os.path.join(attempt_folder_name, train_score_final_file_name)

    def load_files_final_metrics(self):
        train_scores = load_txt(self.train_score_final_file_full_path)
        valid_scores = load_txt(self.valid_score_file_full_path)
        valid_scores = valid_scores[-1]
        valid_scores = split_score_by_labels(valid_scores)


        return train_scores, valid_scores[1][:,1], valid_scores[0][:,1]

    def load_files_over_epochs(self):
        train_scores = load_txt(self.train_score_file_full_path)
        valid_scores = load_txt(self.valid_score_file_full_path)

        valid_scores = mean_labels_over_epochs(valid_scores)

        return train_scores, valid_scores['Class_0'], valid_scores['Class_1']


    def estimate_threshold(self):
        train_scores, valid_scores_class_0, valid_scores_class_1 = self.load_files_final_metrics()
        train_over_epoch, valid_over_epoch_class_0, valid_over_epoch_class_1 = self.load_files_over_epochs()

        plt.figure()
        plt.plot(train_scores)
        plt.plot(valid_scores_class_0)
        plt.plot(valid_scores_class_1)
        plt.show()



