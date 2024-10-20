import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.utils.extmath import density
from Framework.postprocessors.postprocesor_general import PostprocessorGeneral
from typing import Callable
from copy import copy

class PDFComparator(PostprocessorGeneral):
    def __init__(self):
        super().__init__()

    @staticmethod
    def norm_scores(data_to_norm):
        final_scores_norm = copy(data_to_norm)
        final_scores_norm -= np.mean(final_scores_norm)
        final_scores_norm /= np.std(final_scores_norm)

        return final_scores_norm


    def estimate_pdf(self,
                     data,
                     kernel_type = 'exponential',
                     bandwidth: float = 0.2,
                     min_max: int = 6):
        kde = KernelDensity(kernel='exponential', bandwidth=bandwidth).fit(data)
        x_d = (np.linspace(-min_max, min_max, 1000)).reshape(-1, 1)
        density = np.exp(kde.score_samples(x_d))

        return x_d, density

    def estimate_pdf_on_valid(self, bandwidth: float = 0.2, min_max: int = 6):
        valid_scores_class_0, valid_scores_class_1 = self.load_and_parse_valid_per_batch_per_epoch()
        valid_scores_class_0 = np.asarray(valid_scores_class_0).reshape(-1,1)
        valid_scores_class_1 = np.asarray(valid_scores_class_1).reshape(-1,1)

        valid_scores_class_0_norm = self.norm_scores(valid_scores_class_0)
        valid_scores_class_1_norm = self.norm_scores(valid_scores_class_1)

        x_d_class_0, density_class_0 = self.estimate_pdf(valid_scores_class_0_norm)
        x_d_class_1, density_class_1 = self.estimate_pdf(valid_scores_class_1_norm)

        return [x_d_class_0, density_class_0], [x_d_class_1, density_class_1]


    def estimate_pdf_on_train(self, bandwidth: float = 0.2, min_max: int = 6):
        train_final_scores, _ = self.load_files_final_metrics()
        train_final_scores = np.asarray(train_final_scores).reshape(-1,1)
        train_final_scores_norm = self.norm_scores(train_final_scores)

        x_d, density = self.estimate_pdf(train_final_scores_norm)


        plt.figure()
        counts0, bin_edges0, _ = plt.hist(train_final_scores_norm, bins=15, density=False, alpha=0.5, label='Histogram of Anomaly Scores',
                 color='orange')

        bin_width = bin_edges0[1] - bin_edges0[0]
        normalized_counts = counts0 / (train_final_scores.shape[0] * bin_width)

        plt.figure()
        plt.bar(bin_edges0[:-1], normalized_counts, width=bin_width, alpha=0.5, label='Normalized Histogram')
        plt.plot(x_d,density, label='KDE', color='red')
        plt.title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        return x_d, density


    def calculate_classification_score(self,
                                       testing_loop: Callable):

        self.estimate_pdf_on_train()
        self.estimate_pdf_on_valid()

        classification_score_valid = None
        classification_score_on_test = None
        fig = None

        return classification_score_valid, classification_score_on_test, [fig]
