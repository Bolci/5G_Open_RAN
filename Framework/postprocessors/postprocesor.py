import os
from Framework.utils.utils import load_txt
from main_postprocessing import valid_scores, train_scores
from Framework.postprocessors.postprocessor_functions import split_score_by_labels


class Postprocessor:
    def __init__(self):
        self.result_folder_path = None
        self.train_score_file_full_path = None
        self.valid_score_file_full_path = None

    def set_paths(self,
                  result_folder_path: str,
                  attempt_name: str,
                  train_score_paths_file_name: str,
                  valid_score_path_file_name: str):
        attempt_folder_name = os.path.join(result_folder_path, attempt_name)

        self.result_folder_path = attempt_folder_name
        self.train_score_file_full_path = os.path.join(attempt_folder_name, train_score_paths_file_name)
        self.valid_score_file_full_path = os.path.join(attempt_folder_name, valid_score_path_file_name)

    def prepate_files(self):
        train_scores = load_txt(self.train_score_file_full_path)
        valid_scores = load_txt(self.valid_score_file_full_path)
        valid_class_0, valid_class_1 = split_score_by_labels(valid_scores)

        return train_scores, valid_class_0, valid_class_1


    def estimate_threshold(self):
        train_scores, valid_class_0, valid_class_1 = self.prepate_files()

