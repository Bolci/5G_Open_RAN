import numpy as np
import torch
from .data_preprocessor_functions import DataPreprocessorFunctions
import os
from ..exceptions.exceptions import DataProcessorException
from ..utils.utils import load_mat_file
from pathlib import Path


class PreprocessorTypes:
    @staticmethod
    def abs_only(original_sequence, raw_data):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data = np.abs(data)
        data =  DataPreprocessorFunctions.to_log(data)
        data = DataPreprocessorFunctions.split_by_groups_of_n(data, 2)
        return data

    @staticmethod
    def abs_only_by_one(original_sequence, raw_data):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data = np.abs(data)
        data = DataPreprocessorFunctions.to_log(data)
        data = DataPreprocessorFunctions.split_by_groups_of_n(data, 1)
        return data

    @staticmethod
    def abs_only_mean_by_group(original_sequence, raw_data):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data = np.abs(data)
        data = DataPreprocessorFunctions.to_log(data)
        data = DataPreprocessorFunctions.mean_by_quaters_axis_2(data)
        data = data[..., np.newaxis]
        data = np.transpose(data, (1, 0, 2))
        return data

    @staticmethod
    def abs_and_phase(original_sequence, raw_data):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data_abs = np.abs(data)
        data_abs = DataPreprocessorFunctions.to_log(data_abs)
        data_abs = DataPreprocessorFunctions.split_by_groups_of_n(data_abs, 2)

        data_phase = np.arctan2(data)
        data_phase = DataPreprocessorFunctions.split_by_groups_of_n(data_phase, 2)

        return np.concatenate((data_abs, data_phase), axis=1)

    @staticmethod
    def raw_IQ(original_sequence, raw_data):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data_real = data.real
        data_imag = data.imag
        data_real = DataPreprocessorFunctions.split_by_groups_of_n(data_real, 2)
        data_imag = DataPreprocessorFunctions.split_by_groups_of_n(data_imag, 2)

        return data_real, data_imag


class DataPreprocessor(PreprocessorTypes):
    def __init__(self):
        super().__init__()
        self.data_cache_path = None
        self.original_seq = []
        self.possible_preprocessing = {'abs_only': lambda x, y: PreprocessorTypes.abs_only(x, y),
                                       'abs_only_mean_by_group': lambda x, y: PreprocessorTypes.abs_only_mean_by_group(x, y),
                                       'abs_only_by_one_sample': lambda x, y: PreprocessorTypes.abs_only_by_one(x, y)}
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
        # full_path = os.path.join(self.data_cache_path, saving_folder_name)
        full_path = saving_folder_name
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
                          full_saving_path: str,
                          merge_files: bool = False) -> None:

        all_files = os.listdir(source_folder_path)
        if merge_files:
            single_matrix = None
            for single_file_name in all_files:
                single_file_path = os.path.join(source_folder_path, single_file_name)
                loaded_file = np.load(single_file_path)
                preprocessed_data = self.possible_preprocessing[preprocessing_type](self.original_seq, loaded_file)

                if single_matrix is None:
                    single_matrix = preprocessed_data
                else:
                    single_matrix = np.concatenate((single_matrix, preprocessed_data), axis=0)

            file_path = os.path.join(full_saving_path,  f"file_{self.counters[data_type]}_label={label}.pt")
            single_processed_data_torch = torch.Tensor(single_matrix)
            torch.save(single_processed_data_torch, file_path)

        else:
            for single_file_name in all_files:
                single_file_path = os.path.join(source_folder_path, single_file_name)
                loaded_file = np.load(single_file_path)
                preprocessed_data = self.possible_preprocessing[preprocessing_type](self.original_seq, loaded_file)

                for single_processed_data in preprocessed_data:

                    file_path = os.path.join(full_saving_path,  f"file_{self.counters[data_type]}_label={label}.pt")
                    single_processed_data_torch = torch.Tensor(single_processed_data)
                    torch.save(single_processed_data_torch, file_path)

                    self.counters[data_type] += 1




    def preprocess_data(self, data_paths: dict,
                        preprocessing_type: str,
                        mix_valid: bool = True,
                        mix_test: bool = True,
                        rewrite_data: bool = False,
                        merge_files: bool = False) -> dict:

        if self.data_cache_path == None:
            raise DataProcessorException("Data cache path is not set")

        if len(self.original_seq) == 0:
            raise DataProcessorException("Original sequence is not set")


        for data_type, list_path in data_paths.items():
            for single_path in list_path:
                measurement_folder = Path(single_path).parts[-1]
                label = self.get_label(measurement_folder)

                mix_bool = False
                if data_type == 'Valid' and mix_valid:
                    mix_bool = True

                if data_type == 'Test' and mix_test:
                    mix_bool = True

                if not mix_bool:
                    path_data_folder = os.path.join(self.data_cache_path, preprocessing_type, data_type,
                                                    measurement_folder)
                else:
                    path_data_folder = os.path.join(self.data_cache_path, preprocessing_type, data_type)

                full_data_path = self.prepare_saving_path(path_data_folder)

                if not full_data_path in self.paths_for_datasets[data_type]:
                    self.paths_for_datasets[data_type].append(full_data_path)

                if rewrite_data:
                    self.preprocess_folder(data_type=data_type,
                                           source_folder_path=single_path,
                                           preprocessing_type=preprocessing_type,
                                           label=label,
                                           full_saving_path= full_data_path,
                                           merge_files=merge_files)

        return self.paths_for_datasets
