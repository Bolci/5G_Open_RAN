import numpy as np
import os
from Framework.postprocessors.postprocesor_general import PostprocessorGeneral
import matplotlib.pyplot as plt


class IntervalEstimatorMinMax(PostprocessorGeneral):
    def __init__(self):
        super().__init__()

    def metric_function_interval(self, loss_of_sample_x):
        return 0 if self.measured_decision_line['min'] <= loss_of_sample_x <= self.measured_decision_line['max'] else 1


    def estimate_decision_lines(self,
                                use_epochs: int = 5,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = False,
                                save_figs: bool = False,
                                figs_label: str = "",
                                ):
        train_scores, valid_scores = self.load_files_final_metrics()
        valid_scores = np.asarray(valid_scores)[:,1]

        val_min, val_max = np.min(valid_scores), np.max(valid_scores)

        self.measured_decision_line = {
            'min': val_min,
            'max': val_max,
        }
        return None, None

    def test(self,
             testing_loop,
             use_epochs: int = 5,
             no_steps_to_estimate: int = 200,
             prepare_figs: bool = False,
             save_figs: bool = False,
             figs_label: str = "",
             ):
        score, predictions = testing_loop(self.metric_function_interval)

        return score


class IntervalEstimatorStd(PostprocessorGeneral):
    def __init__(self):
        super().__init__()
        self.std_val_threshold = 1
    
    def metric_function_std(self, loss_of_sample_x):
        _min = self.measured_decision_line['mean'] - self.measured_decision_line['std']*self.std_val_threshold
        _max = self.measured_decision_line['mean'] + self.measured_decision_line['std']*self.std_val_threshold

        return 0 if _min <= loss_of_sample_x <= _max else 1

    def estimate_decision_lines(self,
                                use_epochs: int = 5,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = False,
                                save_figs: bool = False,
                                figs_label: str = "",
                                ):
        train_scores, valid_scores = self.load_files_final_metrics()
        valid_scores = np.asarray(valid_scores)[:,1]

        val_mean, val_std = np.mean(valid_scores), np.std(valid_scores)

        self.measured_decision_line = {
            'mean': val_mean,
            'std': val_std,
        }
        return None, None

    def std_diagram(self, testing_loop, boundaries = (1, 10)):
        ranges = np.linspace(*boundaries, 19)

        scores = []
        for std_val in ranges:
            self.std_val_threshold = std_val
            score, predictions = testing_loop(self.metric_function_std)
            scores.append(score)

        return ranges, scores


    def test(self,
             testing_loop,
             use_epochs: int = 5,
             no_steps_to_estimate: int = 200,
             prepare_figs: bool = False,
             save_figs: bool = False,
             figs_label: str = "",
             ):
        #score, predictions = testing_loop(self.metric_function_std)

        ranges, scores = self.std_diagram(testing_loop)
        max_id = np.argmax(scores)

        max_sigma = ranges[max_id]
        max_score = scores[max_id]

        if save_figs:
            fig, ax = plt.subplots()
            ax.vlines(max_sigma, ymin=0, ymax=1, linestyles='dashed', alpha=0.5, color='blue',
                      label='max_sigma', linewidth=2.0)
            ax.plot(ranges, scores, linewidth=2.0)
            ax.plot(max_sigma, max_score, 'r*')
            ax.grid()
            ax.set_ylim([0, 1])
            ax.set_xlim([ranges[0], ranges[-1]])
            ax.set_title(
                f"Classification score = {max_score:.4f}, \n Sigma={max_sigma:.2f} on validation data")
            ax.set_xlabel('Sigma')
            ax.set_ylabel('Classification score [%]')

            if save_figs:
                saving_path = os.path.join(self.result_folder_path, f'interval_estimator_{figs_label}.png')
                fig.savefig(saving_path)
            plt.close(fig)

        return max_score
    
    