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
        # kernel options = 'linear', 'cosine', 'epanechnikov', 'tophat', 'gaussian', 'exponential'
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
        x_d = (np.linspace(-min_max, min_max, 1000)).reshape(-1, 1)
        density = np.exp(kde.score_samples(x_d))

        return x_d, density

    @staticmethod
    def normalize_hist_counts(score_norm, bins=15):
        fig_hist, ax_hist = plt.subplots()
        counts0, bin_edges0, _ = ax_hist.hist(score_norm, bins=bins, density=False)
        plt.close(fig_hist)

        bin_width = bin_edges0[1] - bin_edges0[0]
        normalized_counts = counts0 / (score_norm.shape[0] * bin_width)

        return bin_width, normalized_counts, bin_edges0


    def estimate_pdf_on_valid(self, bandwidth: float = 0.2, min_max: int = 6):
        valid_scores_class_0, valid_scores_class_1 = self.load_and_parse_valid_per_batch_per_epoch()
        valid_scores_class_0 = np.asarray(valid_scores_class_0).reshape(-1,1)
        valid_scores_class_1 = np.asarray(valid_scores_class_1).reshape(-1,1)

        valid_scores_class_0_norm = self.norm_scores(valid_scores_class_0)
        valid_scores_class_1_norm = self.norm_scores(valid_scores_class_1)

        x_d_class_0, density_class_0 = self.estimate_pdf(valid_scores_class_0_norm)
        x_d_class_1, density_class_1 = self.estimate_pdf(valid_scores_class_1_norm)

        bin_width, normalized_counts_class_0, bin_edges0 = self.normalize_hist_counts(density_class_0)
        bin_width, normalized_counts_class_1, bin_edges0 = self.normalize_hist_counts(density_class_1)

        return [x_d_class_0, density_class_0], [x_d_class_1, density_class_1]


    def estimate_pdf_on_train(self, bandwidth: float = 0.2, min_max: int = 6):
        train_final_scores, _ = self.load_files_final_metrics()
        train_final_scores = np.asarray(train_final_scores).reshape(-1,1)
        train_final_scores_norm = self.norm_scores(train_final_scores)
        x_d, density = self.estimate_pdf(train_final_scores_norm)

        bin_width, normalized_counts, bin_edges0 = self.normalize_hist_counts(train_final_scores_norm)

        fig, ax = plt.subplots()
        ax.bar(bin_edges0[:-1], normalized_counts, width=bin_width, alpha=0.5, label='Normalized Histogram')
        ax.plot(x_d,density, label='KDE', color='red')
        ax.set_title('DCAE PDF Approach: Estimated PDF of Anomaly Scores')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

        return [x_d, density], [fig]

    def estimate_decision_lines(self, *args, **kwargs):
        return None, None


    def test(self,
                                       testing_loop: Callable):

        #train_values, figs = self.estimate_pdf_on_train()
        #valid_values_class_0, valid_values_class_1 = self.estimate_pdf_on_valid()


        classification_score_valid = None
        classification_score_on_test = None
        fig = None

        return classification_score_valid, classification_score_on_test, [fig]
