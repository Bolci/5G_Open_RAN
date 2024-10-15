import os
import numpy as np
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


    def estimate_threshold(self, use_epochs:int = 5, no_steps_to_estimate:int = 200, save_into_fig:bool = True):
        train_scores, valid_scores_class_0, valid_scores_class_1 = self.load_files_final_metrics()
        train_over_epoch, valid_over_epoch_class_0, valid_over_epoch_class_1 = self.load_files_over_epochs()

        valid_scores_class_0 = np.asarray(valid_scores_class_0)
        valid_scores_class_1 = np.asarray(valid_scores_class_1)

        no_all_valid_class_0 = valid_scores_class_0.shape[0]
        no_all_valid_class_1 = valid_scores_class_1.shape[0]
        no_all_valid = no_all_valid_class_0 + no_all_valid_class_1

        train_scores = np.mean(np.asarray(train_over_epoch[-use_epochs:]))
        valid_scores_0 = np.mean(np.asarray(valid_over_epoch_class_0[-use_epochs:]))
        valid_scores_1 = np.mean(np.asarray(valid_over_epoch_class_1[-use_epochs:]))

        all_scores = np.asarray([train_scores, valid_scores_0, valid_scores_1])
        max_score = np.max(all_scores)
        min_score = np.min(all_scores)

        max_score += 0.2*max_score
        min_score -= 0.2*min_score

        ds = (max_score - min_score)/no_steps_to_estimate

        boundary_scores = []
        no_class_all = []

        for x in range(200):
            boundary_score = min_score + ds*x

            no_class_0 = len(np.where(boundary_score <= valid_scores_class_0)[0])
            no_class_1 = len(np.where(boundary_score > valid_scores_class_1)[0])

            no_class_all.append((no_class_1 + no_class_0)/(no_all_valid_class_0+no_all_valid_class_1))
            boundary_scores.append(boundary_score)

        boundary_scores = np.asarray(boundary_scores)
        no_class_all = np.asarray(no_class_all)

        idx_max = np.argmax(no_class_all)
        threshold = boundary_scores[idx_max]
        classification_score = no_class_all[idx_max]

        if save_into_fig:
            plt.figure()
            plt.vlines(boundary_scores[idx_max], ymin=0, ymax=1, linestyles='dashed', alpha=0.5)
            plt.plot(boundary_scores, no_class_all)
            plt.plot(threshold,classification_score , 'r*')
            plt.grid()
            plt.ylim([0,1])
            plt.xlim([boundary_scores[0], boundary_scores[-1]])
            plt.title(f"Classification score = {classification_score:.4f}, \n Threshold={threshold:.8f} on validartion data")
            saving_path = os.path.join(self.result_folder_path, 'valid_data_threshold_estimation.png')
            plt.xlabel('Metrics score')
            plt.ylabel('Classification score [%]')
            plt.savefig(saving_path)


        return threshold, classification_score

    def estimate_PDF(self):
        pass
