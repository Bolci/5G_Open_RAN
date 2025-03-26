import numpy as np
import os
from Framework.postprocessors.postprocesor_general import PostprocessorGeneral
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Any


class IntervalEstimatorMinMax(PostprocessorGeneral):
    """
    A class for estimating decision intervals based on the minimum and maximum scores from the validation data.

    Inherits from the `PostprocessorGeneral` class and provides methods for defining the decision interval
    based on the minimum and maximum values of validation scores. It also includes functionality for testing
    and evaluating the model based on this decision interval.

    Attributes:
        measured_decision_line (dict): Contains 'min' and 'max' values, representing the decision boundaries.

    Methods:
        metric_function_interval(loss_of_sample_x):
            Evaluates whether a given sample falls within the defined decision boundaries.

        estimate_decision_lines(use_epochs, no_steps_to_estimate, prepare_figs, save_figs, figs_label):
            Estimates the decision boundaries based on validation data.

        test(testing_loop, use_epochs, no_steps_to_estimate, prepare_figs, save_figs, figs_label):
            Tests the model using the defined decision boundaries and calculates the classification score.
    """

    def __init__(self):
        super().__init__()

    def metric_function_interval(self, loss_of_sample_x: float) -> int:
        """
       Evaluates whether the sample is within the defined decision interval. Server as a metric function for the test loop

       Args:
           loss_of_sample_x (float): The score of the sample to evaluate.

       Returns:
           int: 0 if the sample is within the interval, 1 otherwise.
       """
        return 0 if self.measured_decision_line['min'] <= loss_of_sample_x <= self.measured_decision_line['max'] else 1

    def get_decision_lines(self):
        _min = self.measured_decision_line['min']
        _max = self.measured_decision_line['max']
        return _min, _max

    def estimate_decision_lines(self,
                                use_epochs: int = 5,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = False,
                                save_figs: bool = False,
                                figs_label: str = "",
                                ) -> Tuple[None, None]:
        """
        Estimates the decision lines based on the min and max validation losses.

        Args:
            use_epochs (int): not used, are there just to have consistent code with other estimators.
            no_steps_to_estimate (int): not used, are there just to have consistent code with other estimators.
            prepare_figs (bool): not used, are there just to have consistent code with other estimators.
            save_figs (bool): not used, are there just to have consistent code with other estimators.
            figs_label (str): not used, are there just to have consistent code with other estimators.

        Returns:
            None
        """
        train_scores, valid_scores = self.load_files_final_metrics()
        valid_scores = np.asarray(valid_scores)[:,1]

        val_min, val_max = np.min(valid_scores), np.max(valid_scores)

        self.measured_decision_line = {
            'min': val_min,
            'max': val_max,
        }
        return None, None

    def test(self,
             testing_loop: Callable,
             use_epochs: int = 5,
             no_steps_to_estimate: int = 200,
             prepare_figs: bool = False,
             save_figs: bool = False,
             figs_label: str = "",
             ):
        """
        Tests the model and calculates the classification score based on the decision boundaries.

        Args:
            testing_loop (Callable): The testing loop function to evaluate the model.
            use_epochs (int): not used, is there just to have consistent code with other estimators.
            no_steps_to_estimate (int): not used, is there just to have consistent code with other estimators.
            prepare_figs (bool): not used, is there just to have consistent code with other estimators.
            save_figs (bool): not used, is there just to have consistent code with other estimators.
            figs_label (str): not used, is there just to have consistent code with other estimators.

        Returns:
        """
        score, predictions, metrics = testing_loop(self.metric_function_interval)

        return score, predictions, metrics


class IntervalEstimatorStd(PostprocessorGeneral):
    """
    A class for estimating decision intervals based on the mean and standard deviation of validation scores.

    Inherits from the `PostprocessorGeneral` class and provides methods for defining the decision interval
    based on the mean and standard deviation of validation scores. It also includes functionality for testing
    and evaluating the model based on this decision interval.

    Attributes:
        measured_decision_line (dict): Contains 'mean' and 'std' values, representing the decision boundaries.
        std_val_threshold (float): The threshold multiplier for standard deviation.

    """
    def __init__(self) -> None:
        super().__init__()
        self.std_val_threshold = 1
    
    def metric_function_std(self, loss_of_sample_x: float) -> int:
        """
        Evaluates whether the sample is within the defined decision interval based on the standard deviation.

        Args:
            loss_of_sample_x (float): The score of the sample to evaluate.

        Returns:
            int: 0 if the sample is within the interval, 1 otherwise.
        """
        _min = self.measured_decision_line['mean'] - self.measured_decision_line['std']*self.std_val_threshold
        _max = self.measured_decision_line['mean'] + self.measured_decision_line['std']*self.std_val_threshold

        return 0 if _min <= loss_of_sample_x <= _max else 1

    def estimate_decision_lines(self,
                                use_epochs: int = 5,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = False,
                                save_figs: bool = False,
                                figs_label: str = "",
                                ) -> Tuple[None, None]:
        """
       Estimates the decision lines based on the mean and standard deviation of validation scores.

       Args:
           use_epochs (int): not used, is there just to have consistent code with other estimators.
           no_steps_to_estimate (int): not used, is there just to have consistent code with other estimators.
           prepare_figs (bool): not used, is there just to have consistent code with other estimators.
           save_figs (bool): not used, is there just to have consistent code with other estimators.
           figs_label (str): not used, is there just to have consistent code with other estimators.

       Returns:
           None
       """

        train_scores, valid_scores = self.load_files_final_metrics()
        valid_scores = np.asarray(valid_scores)[:,1]

        val_mean, val_std = np.mean(valid_scores), np.std(valid_scores)

        self.measured_decision_line = {
            'mean': val_mean,
            'std': val_std,
        }
        return None, None

    def get_decision_lines(self):
        _min = self.measured_decision_line['mean'] - self.measured_decision_line['std']*self.std_val_threshold
        _max = self.measured_decision_line['mean'] + self.measured_decision_line['std']*self.std_val_threshold
        return _min, _max


    def std_diagram(self,
                    testing_loop: Callable,
                    boundaries: Tuple[int, int] = (1, 10)) -> Tuple[np.array, list, list]:
        """
        Creates a diagram of classification scores over different standard deviation thresholds.

        Args:
            testing_loop (function): The testing loop function to evaluate the model.
            boundaries (tuple): Range of standard deviation values to evaluate.

        Returns:
            # tuple: Ranges of standard deviation values and corresponding classification scores.
        """

        ranges = np.linspace(*boundaries, 19)

        scores = []
        predictions = []
        metrics = []
        for std_val in ranges:
            self.std_val_threshold = std_val
            score, predictions, _metrics = testing_loop(self.metric_function_std)
            scores.append(score)
            metrics.append(_metrics)

        return ranges, scores, predictions, metrics


    def test(self,
             testing_loop,
             use_epochs: int = 5,
             no_steps_to_estimate: int = 200,
             prepare_figs: bool = False,
             save_figs: bool = False,
             figs_label: str = "",
             ):
        """
           Tests the model and calculates the classification score based on the decision boundaries.

           Args:
               testing_loop (function): The testing loop function to evaluate the model.
               use_epochs (int): not used, is there just to have consistent code with other estimators.
               no_steps_to_estimate (int): not used, is there just to have consistent code with other estimators.
               prepare_figs (bool): Whether to prepare figures for visualization.
               save_figs (bool): Whether to save figures after visualization.
               figs_label (str): Label for saving the figures.

           Returns:

        """

        score, predictions, metrics = testing_loop(self.metric_function_std)
        return score, predictions, metrics
    

class IntervalEstimatorMAD(PostprocessorGeneral):
    """
    A class for estimating decision intervals based on the median and median absolute deviation (MAD) of validation scores.

    Inherits from the `PostprocessorGeneral` class and provides methods for defining the decision interval
    based on the median and MAD of validation scores. It also includes functionality for testing
    and evaluating the model based on this decision interval.

    Attributes:
        measured_decision_line (dict): Contains 'median' and 'mad' values, representing the decision boundaries.
        mad_val_threshold (float): The threshold multiplier for the MAD.
    """
    def __init__(self) -> None:
        super().__init__()
        self.mad_val_threshold = 1.0

    def metric_function_mad(self, loss_of_sample_x: float) -> int:
        """
        Evaluates whether the sample is within the defined decision interval based on MAD.

        Args:
            loss_of_sample_x (float): The score of the sample to evaluate.

        Returns:
            int: 0 if the sample is within the interval, 1 otherwise.
        """
        median = self.measured_decision_line['median']
        mad = self.measured_decision_line['mad']
        _min = median - mad * self.mad_val_threshold
        _max = median + mad * self.mad_val_threshold
        return 0 if _min <= loss_of_sample_x <= _max else 1


    def get_decision_lines(self):
        _min = self.measured_decision_line['median'] - self.measured_decision_line['mad']*self.mad_val_threshold
        _max = self.measured_decision_line['median'] + self.measured_decision_line['mad']*self.mad_val_threshold
        return _min, _max

    def estimate_decision_lines(self,
                                use_epochs: int = 5,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = False,
                                save_figs: bool = False,
                                figs_label: str = "",
                                ) -> Tuple[None, None]:
        """
        Estimates the decision lines based on the median and MAD of validation scores.

        Args:
            use_epochs (int): Not used, provided for consistency with other estimators.
            no_steps_to_estimate (int): Not used, provided for consistency with other estimators.
            prepare_figs (bool): Not used, provided for consistency with other estimators.
            save_figs (bool): Not used, provided for consistency with other estimators.
            figs_label (str): Not used, provided for consistency with other estimators.

        Returns:
            Tuple[None, None]
        """
        train_scores, valid_scores = self.load_files_final_metrics()
        valid_scores = np.asarray(valid_scores)[:, 1]

        median_val = np.median(valid_scores)
        mad_val = np.median(np.abs(valid_scores - median_val))

        self.measured_decision_line = {
            'median': median_val,
            'mad': mad_val,
        }
        return None, None

    def mad_diagram(self,
                    testing_loop: Callable,
                    boundaries: Tuple[float, float] = (0.5, 10.0)) -> Tuple[np.array, list, list, list]:
        """
        Creates a diagram of classification scores over different MAD threshold multipliers.

        Args:
            testing_loop (Callable): The testing loop function to evaluate the model.
            boundaries (Tuple[float, float]): The range of MAD multiplier values to evaluate.

        Returns:
            Tuple containing:
                - ranges (np.array): Array of MAD multiplier values.
                - scores (list): Corresponding classification scores.
                - predictions (list): Predictions for each threshold.
                - metrics (list): Metrics for each threshold.
        """
        ranges = np.linspace(*boundaries, 19)
        scores = []
        predictions = []
        metrics = []
        for mad_val in ranges:
            self.mad_val_threshold = mad_val
            score, preds, _metrics = testing_loop(self.metric_function_mad)
            scores.append(score)
            predictions.append(preds)
            metrics.append(_metrics)

        return ranges, scores, predictions, metrics

    def test(self,
             testing_loop: Callable,
             use_epochs: int = 5,
             no_steps_to_estimate: int = 200,
             prepare_figs: bool = False,
             save_figs: bool = False,
             figs_label: str = "",
             ):
        """
        Tests the model and calculates the classification score based on the MAD-based decision boundaries.

        Args:
            testing_loop (Callable): The testing loop function to evaluate the model.
            use_epochs (int): Not used, provided for consistency with other estimators.
            no_steps_to_estimate (int): Not used, provided for consistency with other estimators.
            prepare_figs (bool): Whether to prepare figures for visualization.
            save_figs (bool): Whether to save figures after visualization.
            figs_label (str): Label for saving the figures.

        Returns:
            Tuple containing the maximum classification score, predictions, and metrics corresponding to the best MAD multiplier.
        """
        score, predictions, metrics = testing_loop(self.metric_function_mad)
        return score, predictions, metrics


