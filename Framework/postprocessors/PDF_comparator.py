import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.utils.extmath import density
from Framework.postprocessors.postprocesor_general import PostprocessorGeneral
from typing import Callable
from copy import copy

class PDFComparator(PostprocessorGeneral):
    """
    A class for comparing Probability Density Functions (PDFs) using Kernel Density Estimation (KDE).
    Inherits from PostprocessorGeneral.
    """

    def __init__(self):
        """
        Initializes the PDFComparator class.
        """
        super().__init__()

    @staticmethod
    def norm_scores(data_to_norm):
        """
        Normalizes the scores by subtracting the mean and dividing by the standard deviation.

        Args:
            data_to_norm (np.ndarray): The data to normalize.

        Returns:
            np.ndarray: The normalized data.
        """
        final_scores_norm = copy(data_to_norm)
        final_scores_norm -= np.mean(final_scores_norm)
        final_scores_norm /= np.std(final_scores_norm)

        return final_scores_norm

    def estimate_pdf(self,
                     data,
                     kernel_type='exponential',
                     bandwidth: float = 0.2,
                     min_max: int = 6):
        """
        Estimates the PDF of the given data using Kernel Density Estimation (KDE).

        Args:
            data (np.ndarray): The data to estimate the PDF for.
            kernel_type (str): The type of kernel to use for KDE.
            bandwidth (float): The bandwidth parameter for KDE.
            min_max (int): The range for the x-axis.

        Returns:
            tuple: A tuple containing the x-axis values and the estimated density.
        """
        # kernel options = 'linear', 'cosine', 'epanechnikov', 'tophat', 'gaussian', 'exponential'
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
        x_d = (np.linspace(-min_max, min_max, 1000)).reshape(-1, 1)
        density = np.exp(kde.score_samples(x_d))

        return x_d, density

    @staticmethod
    def normalize_hist_counts(score_norm, bins=15):
        """
        Normalizes the histogram counts of the given scores.

        Args:
            score_norm (np.ndarray): The normalized scores.
            bins (int): The number of bins for the histogram.

        Returns:
            tuple: A tuple containing the bin width, normalized counts, and bin edges.
        """
        fig_hist, ax_hist = plt.subplots()
        counts0, bin_edges0, _ = ax_hist.hist(score_norm, bins=bins, density=False)
        plt.close(fig_hist)

        bin_width = bin_edges0[1] - bin_edges0[0]
        normalized_counts = counts0 / (score_norm.shape[0] * bin_width)

        return bin_width, normalized_counts, bin_edges0

    def estimate_pdf_on_valid(self, bandwidth: float = 0.2, min_max: int = 6):
        """
        Estimates the PDF on validation data.

        Args:
            bandwidth (float): The bandwidth parameter for KDE.
            min_max (int): The range for the x-axis.

        Returns:
            list: A list containing the x-axis values and the estimated density for class 0 and class 1.
        """
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
        """
        Estimates the PDF on training data.

        Args:
            bandwidth (float): The bandwidth parameter for KDE.
            min_max (int): The range for the x-axis.

        Returns:
            list: A list containing the x-axis values, the estimated density, and the figure.
        """
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
        """
        Estimates the decision lines. Currently not implemented.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing None, None.
        """
        return None, None

    def test(self,
             testing_loop = None,
             use_epochs = None,
             no_steps_to_estimate = None,
             prepare_figs = None,
             save_figs = None,
             figs_label = None):
        """
        Tests the PDFComparator. Currently not implemented.

        Args:
            testing_loop: The testing loop.
            use_epochs: Whether to use epochs.
            no_steps_to_estimate: The number of steps to estimate.
            prepare_figs: Whether to prepare figures.
            save_figs: Whether to save figures.
            figs_label: The label for the figures.

        Returns:
            tuple: A tuple containing classification_score_valid, classification_score_on_test, and a list of figures.
        """
        #train_values, figs = self.estimate_pdf_on_train()
        #valid_values_class_0, valid_values_class_1 = self.estimate_pdf_on_valid()

        classification_score_valid = None
        classification_score_on_test = None
        fig = None

        return classification_score_valid, classification_score_on_test, [fig]
