from pstats import func_strip_path

import numpy as np
import torch

from .data_preprocessor_functions import DataPreprocessorFunctions
import os
from ..exceptions.exceptions import DataProcessorException
from ..utils.utils import load_mat_file


class PreprocessorTypes:
    @staticmethod
    def abs_only(original_sequence, raw_data):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data = np.abs(data)
        return DataPreprocessorFunctions.split_by_groups_of_n(data, 2)

class DataPreprocessor(PreprocessorTypes):
    def __init__(self):
        super().__init__()
        self.data_cache_path = None
        self.original_seq = []
        self.possible_preprocessing = {'abs_only': lambda x, y: PreprocessorTypes.abs_only(x, y)}
        self.counters = {"Train": 0,
                         "Valid": 0,
                         "Test": 0}
        self.paths_for_datasets = {'Train': [], 'Test': [], 'Valid': []}

    def set_cache_path(self, path: str) -> None:
        self.data_cache_path = path

    def set_original_seg(self, path: str) -> None:
        mat_file = load_mat_file(path)
        pss_sss_raw = mat_file['rxGridSSBurst']
        pss_sss_raw = DataPreprocessorFunctions.mean_by_quaters(pss_sss_raw)
        self.original_seq = pss_sss_raw

    def prepare_saving_path(self, saving_folder_name: str) -> str:
        full_path = os.path.join(self.data_cache_path, saving_folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        return full_path

    @staticmethod
    def get_label(criteria: str) -> int:
        label = 0
        if 'wPA' in criteria:
            label = 1
        return label

    def preprocess_folder(self,
                          data_type: str,
                          source_folder_path: str,
                          preprocessing_type:str,
                          label: int,
                          full_saving_path: str):

        all_files = os.listdir(source_folder_path)
        for single_file_name in all_files:
            single_file_path = os.path.join(source_folder_path, single_file_name)
            loaded_file = np.load(single_file_path)
            preprocessed_data = self.possible_preprocessing[preprocessing_type](self.original_seq, loaded_file)

            for single_processed_data in preprocessed_data:

                file_path = os.path.join(full_saving_path,  f"file_{self.counters[data_type]}_label={label}.pt")
                single_processed_data_torch = torch.Tensor(single_processed_data)
                torch.save(single_processed_data_torch, file_path)

                self.counters[data_type] += 1

            break


    def preprocess_data(self, data_paths: dict,
                        preprocessing_type: str,
                        mix_valid: bool = False,
                        mix_test: bool = True,
                        preprocessing_performed: bool = False) -> dict:

        if self.data_cache_path == None:
            raise DataProcessorException("Data cache path is not set")

        if len(self.original_seq) == 0:
            raise DataProcessorException("Original sequence is not set")


        for data_type, list_path in data_paths.items():
            for single_path in list_path:
                measurement_folder = single_path.split('/')[-1]
                label = self.get_label(measurement_folder)

                mix_bool = False
                if data_type == 'Valid' and mix_valid:
                    mix_bool = True

                if data_type == 'Test' and mix_test:
                    mix_bool = True

                if not mix_bool:
                    path_data_folder = os.path.join(self.data_cache_path, preprocessing_type, data_type,
                                                    measurement_folder)
                    full_data_path = self.prepare_saving_path(path_data_folder)
                else:
                    path_data_folder = os.path.join(self.data_cache_path, preprocessing_type, data_type)
                    full_data_path = self.prepare_saving_path(path_data_folder)

                if not full_data_path in self.paths_for_datasets[data_type]:
                    self.paths_for_datasets[data_type].append(full_data_path)

                if not preprocessing_performed:
                    self.preprocess_folder(data_type=data_type,
                                           source_folder_path=single_path,
                                           preprocessing_type=preprocessing_type,
                                           label=label,
                                           full_saving_path= full_data_path,)

        return self.paths_for_datasets
