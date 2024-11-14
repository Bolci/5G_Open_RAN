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
    def abs_only_multichannel(original_sequence, raw_data, max_dim: int = 48):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data_abs = np.abs(data).astype(np.float32)
        data_abs = DataPreprocessorFunctions.to_log(data_abs)

        if data_abs.shape[1] > max_dim:
            data_abs = data_abs[:, :max_dim]

        if data_abs.shape[1] < max_dim:
            zero_data = np.zeros((data_abs.shape[0], 1), dtype=np.float32)

            for x in range(max_dim - data_abs.shape[1]):
                data_abs = np.concatenate((data_abs, zero_data), axis=1)
        data_abs = np.expand_dims(data_abs, axis=0)

        return data_abs

    @staticmethod
    def raw_IQ(original_sequence, raw_data, max_dim: int = 48):
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)

        if data.shape[1] > max_dim:
            data = data[:, :max_dim]

        data_real = data.real
        data_imag = data.imag
        result = np.empty((data_real.shape[0],data_real.shape[1]*2), dtype=data_real.dtype)
        result[:, 0::2] = data_real
        result[:, 1::2] = data_imag

        result = np.expand_dims(result, axis=0)

        return result

    def __init__(self):
        self.data_cache_path = None
        self.original_seq = []
                                       'abs_only_mean_by_group': lambda x, y: PreprocessorTypes.abs_only_mean_by_group(x, y),
                                       'abs_only_by_one_sample': lambda x, y: PreprocessorTypes.abs_only_by_one(x, y),
                                       'abs_only_multichannel': lambda x, y: PreprocessorTypes.abs_only_multichannel(x, y),
                                       'raw_IQ': lambda x, y: PreprocessorTypes.raw_IQ(x, y),
                                       }
        self.paths_for_datasets = {'Train': [], 'Test': [], 'Valid': []}


    def set_cache_path(self, path: str) -> None:
        self.data_cache_path = path

    def set_original_seg(self, path: str) -> None:
        mat_file = load_mat_file(path)
        pss_sss_raw = mat_file['rxGridSSBurst']
        pss_sss_raw = DataPreprocessorFunctions.mean_by_quaters(pss_sss_raw)
        self.original_seq = pss_sss_raw

    def prepare_saving_path(self, saving_folder_name: str) -> str:

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

    @staticmethod

                         data_type: str,
                         preprocessing_type:str,
                         label: int,
                         full_saving_path: str,
                         merge_files: bool = False) -> None:

        if merge_files:
            single_matrix = None
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
                loaded_file = np.load(single_file_path)
                preprocessed_data = self.possible_preprocessing[preprocessing_type](self.original_seq, loaded_file)

                for single_processed_data in preprocessed_data:
                    file_path = os.path.join(full_saving_path,  f"file_{self.counters[data_type]}_label={label}.pt")
                    single_processed_data_torch = torch.Tensor(single_processed_data)
                    torch.save(single_processed_data_torch, file_path)

                    self.counters[data_type] += 1


                          preprocessing_type:str,
                          merge_files: bool = False,




        else:



        for single_path in list_path:
            measurement_folder = Path(single_path).parts[-1]
            label = self.get_label(measurement_folder)

            if not mix_bool:
                                                data_type,
                                                measurement_folder)
            else:
                                                data_type)

            full_data_path = self.prepare_saving_path(path_data_folder)

                self.paths_for_datasets[data_type].append(full_data_path)

            if rewrite_data:
                self.preprocess_folder(data_type=data_type,
                                       preprocessing_type=preprocessing_type,
                                       label=label,
                                       full_saving_path=full_data_path,
                                       merge_files=merge_files)

        return self.paths_for_datasets