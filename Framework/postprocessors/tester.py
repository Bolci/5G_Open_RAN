from Framework.postprocessors.threshold_estimator import ThresholdEstimator
from Framework.postprocessors.PDF_comparator import PDFComparator
from Framework.postprocessors.interval_estimator import IntervalEstimatorStd, IntervalEstimatorMinMax
from Framework.postprocessors.dsvdd_estimator import DSVDDEstimator
from typing import Callable, List
from copy import copy


class TesterFactory:
    test_options = \
        {
            'threshold_estimator': lambda: ThresholdEstimator(),
            'pdf_comparator': lambda: PDFComparator(),
            'interval_estimator_min_max': lambda: IntervalEstimatorMinMax(),
            'interval_estimator_std': lambda: IntervalEstimatorStd(),
            'DSVDD_estimator': lambda: DSVDDEstimator()
        }

    @staticmethod
    def get_test(test_type):
        if test_type not in TesterFactory.test_options:
            raise ValueError(f"Unknown test type: {test_type}")

        return TesterFactory.test_options[test_type]


class Tester:
    """
        A class to handle testing and estimation of decision lines using different postprocessors.
    """

    def __init__(self,
                 result_folder_path: str,
                 attempt_name: str,
                 train_score_over_epoch_file_name: str,
                 valid_score_over_epoch_file_name: str,
                 valid_score_over_epoch_per_batch_file_name: str,
                 train_score_final_file_name: str,
                 tests_to_perform: List[str] = ('DSVDD_estimator',)):

        self.result_folder_path = result_folder_path
        self.attempt_name = attempt_name
        self.train_score_over_epoch_file_name = train_score_over_epoch_file_name
        self.valid_score_over_epoch_file_name = valid_score_over_epoch_file_name
        self.valid_score_over_epoch_per_batch_file_name = valid_score_over_epoch_per_batch_file_name
        self.train_score_final_file_name = train_score_final_file_name

        self.tester_buffer = {}
        self.prepare_tests(tests_to_perform)


    def prepare_tests(self, test_types: list) -> None:
        for test_type in test_types:
            self.tester_buffer[test_type] = TesterFactory.get_test(test_type)()
            self.tester_buffer[test_type].set_paths(result_folder_path=self.result_folder_path,
                                                    attempt_name=self.attempt_name,
                                                    train_score_over_epoch_file_name=self.train_score_over_epoch_file_name,
                                                    valid_score_over_epoch_file_name=self.valid_score_over_epoch_file_name,
                                                    valid_score_over_epoch_per_batch_file_name=self.valid_score_over_epoch_per_batch_file_name,
                                                    train_score_final_file_name=self.train_score_final_file_name)

    def estimate_decision_lines(self,
                                model,
                                use_epochs: int = 1,
                                no_steps_to_estimate: int = 200,
                                prepare_figs: bool = False,
                                save_figs: bool = False,
                                figs_label: str = "valid_scores_over_threshold"
                                ) -> dict:

        valid_scores = {}
        for single_tester_name, single_tester in self.tester_buffer.items():
            threshold, classification_score = single_tester.estimate_decision_lines(model = model,
                                                                                    use_epochs=use_epochs,
                                                                                    no_steps_to_estimate=no_steps_to_estimate,
                                                                                    save_figs=save_figs,
                                                                                    prepare_figs=prepare_figs,
                                                                                    figs_label=f'{figs_label}_{single_tester_name}')
            valid_scores[single_tester_name] = copy(classification_score)

        return {'Validation scores': valid_scores}

    def test_data(self,
                  testing_loop: Callable,
                  use_epochs: int = 1,
                  no_steps_to_estimate: int = 200,
                  prepare_figs: bool = True,
                  save_figs: bool = True,
                  figs_label: str = "test_scores_over_threshold"):

        testing_scores = {}
        predictions_buffer = {}
        metrics_buffer = {}

        for single_tester_name, single_tester in self.tester_buffer.items():
            classification_score_on_test, predictions, metrics = single_tester.test(testing_loop,
                                                                           use_epochs=use_epochs,
                                                                           no_steps_to_estimate=no_steps_to_estimate,
                                                                           prepare_figs=prepare_figs,
                                                                           save_figs=save_figs,
                                                                           figs_label=figs_label)
            testing_scores[single_tester_name] = copy(classification_score_on_test)
            predictions_buffer[single_tester_name] = predictions
            metrics_buffer[single_tester_name] = metrics

        return testing_scores, predictions_buffer, metrics_buffer