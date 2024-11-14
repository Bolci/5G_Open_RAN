import numpy as np
import torch
from .data_preprocessor_functions import DataPreprocessorFunctions
import os
from ..exceptions.exceptions import DataProcessorException
from ..utils.utils import load_mat_file, chkList, flatten
from pathlib import Path
from typing import Callable
from copy import copy


class PreprocessorTypes:
    """
    Namespace of functions that manage the sequences to preprocess data. Current preprocessings:
    - abs_only
    - abs_only_by_one
    - abs_only_mean_by_group
    - abs_and_phase
    - abs_only_multichannel
    - raw_IQ

    Expected return format: (N, d, C) - number of data, length of data, number of channels of data.
    """
    @staticmethod
    def abs_only(original_sequence, raw_data):
        """
        Preprocess data by taking the absolute value, converting to log scale, and splitting by groups of 2.

        Args:
            original_sequence: The original sequence data.
            raw_data: The raw data to preprocess.

        Returns:
            Preprocessed data.
        """
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data = np.abs(data)
        data = DataPreprocessorFunctions.to_log(data)
        data = DataPreprocessorFunctions.split_by_groups_of_n(data, 2)
        return data

    @staticmethod
    def abs_only_by_one(original_sequence, raw_data):
        """
        Preprocess data by taking the absolute value, converting to log scale, and splitting by groups of 1.

        Args:
            original_sequence: The original sequence data.
            raw_data: The raw data to preprocess.

        Returns:
            Preprocessed data.
        """
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data = np.abs(data)
        data = DataPreprocessorFunctions.to_log(data)
        data = DataPreprocessorFunctions.split_by_groups_of_n(data, 1)
        return data

    @staticmethod
    def abs_only_mean_by_group(original_sequence, raw_data):
        """
        Preprocess data by taking the absolute value, converting to log scale, averaging by quarters, and transposing.

        Args:
            original_sequence: The original sequence data.
            raw_data: The raw data to preprocess.

        Returns:
            Preprocessed data.
        """
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data = np.abs(data)
        data = DataPreprocessorFunctions.to_log(data)
        data = DataPreprocessorFunctions.mean_by_quaters_axis_2(data)
        data = data[..., np.newaxis]
        data = np.transpose(data, (1, 0, 2))
        return data

    @staticmethod
    def abs_and_phase(original_sequence, raw_data):
        """
        Preprocess data by taking the absolute value and phase, converting to log scale, and splitting by groups of 2.

        Args:
            original_sequence: The original sequence data.
            raw_data: The raw data to preprocess.

        Returns:
            Preprocessed data concatenated along the second axis.
        """
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)
        data_abs = np.abs(data)
        data_abs = DataPreprocessorFunctions.to_log(data_abs)
        data_abs = DataPreprocessorFunctions.split_by_groups_of_n(data_abs, 2)

        data_phase = np.arctan2(data)
        data_phase = DataPreprocessorFunctions.split_by_groups_of_n(data_phase, 2)

        return np.concatenate((data_abs, data_phase), axis=1)

    @staticmethod
    def abs_only_multichannel(original_sequence, raw_data, max_dim: int = 48):
        """
        Preprocess multichannel data by taking the absolute value, converting to log scale, and padding/truncating to max_dim.

        Args:
            original_sequence: The original sequence data.
            raw_data: The raw data to preprocess.
            max_dim (int): The maximum dimension for the data.

        Returns:
            Preprocessed data.
        """
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
        """
        Preprocess raw IQ data by separating real and imaginary parts, and padding/truncating to max_dim.

        Args:
            original_sequence: The original sequence data.
            raw_data: The raw data to preprocess.
            max_dim (int): The maximum dimension for the data.

        Returns:
            Preprocessed data.
        """
        data = DataPreprocessorFunctions.estimate_channels(raw_data, original_sequence)

        if data.shape[1] > max_dim:
            data = data[:, :max_dim]

        data_real = data.real
        data_imag = data.imag
        result = np.empty((data_real.shape[0], data_real.shape[1] * 2), dtype=data_real.dtype)
        result[:, 0::2] = data_real
        result[:, 1::2] = data_imag

        result = np.expand_dims(result, axis=0)

        return result


class DataPreprocessor(PreprocessorTypes):
    def __init__(self):
        super().__init__()
        self.data_cache_path = None
        self.original_seq = []
        self.possible_preprocessing = {'abs_only': lambda x, y: PreprocessorTypes.abs_only(x, y),
                                       'abs_only_mean_by_group': lambda x, y: PreprocessorTypes.abs_only_mean_by_group(x, y),
                                       'abs_only_by_one_sample': lambda x, y: PreprocessorTypes.abs_only_by_one(x, y),
                                       'abs_only_multichannel': lambda x, y: PreprocessorTypes.abs_only_multichannel(x, y),
                                       'raw_IQ': lambda x, y: PreprocessorTypes.raw_IQ(x, y),
                                       }
        self.counters = {"Train": 0,
                         "Valid": 0,
                         "Test": 0}

        #self.paths_for_datasets = {'Train': [], 'Test': [], 'Valid': []}

        self.buffers_train = {'source_paths': [], 'saving_paths' : [], 'labels':[]}
        self.buffers_valid = {'source_paths': [], 'saving_paths' : [], 'labels':[]}
        self.buffers_test  = {'source_paths': [], 'saving_paths' : [], 'labels':[]}


    def set_cache_path(self, path: str) -> None:
        """
        Set the cache path for storing preprocessed data.

        Args:
            path (str): The path to set as the cache path.
        """
        self.data_cache_path = path

    def set_original_seg(self, path: str) -> None:
        """
        Set the original sequence from a .mat file.

        Args:
            path (str): The path to the .mat file.
        """
        mat_file = load_mat_file(path)
        pss_sss_raw = mat_file['rxGridSSBurst']
        pss_sss_raw = DataPreprocessorFunctions.mean_by_quaters(pss_sss_raw)
        self.original_seq = pss_sss_raw

    @staticmethod
    def prepare_saving_path(saving_folder_name: str) -> None:
        """
        Prepare the saving path by creating the directory if it does not exist.

        Args:
            saving_folder_name (str): The name of the folder to save data in.

        Returns:
            str: The full path to the saving folder.
        """
        # full_path = os.path.join(self.data_cache_path, saving_folder_name)
        full_path = saving_folder_name
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    @staticmethod
    def get_label(criteria: str) -> int:
        """
        Get the label based on the criteria.

        Args:
            criteria (str): The criteria to determine the label.

        Returns:
            int: The label (1 if 'wPA' in criteria, else 0).
        """
        label = 0
        if 'wPA' in criteria:
            label = 1
        return label

    @staticmethod
    def prepare_full_paths(path, files):
        return [os.path.join(path, x) for x in files]


    def preprocess_folder(self,
                          data_type,
                          buffer,
                          preprocessing_type,
                          merge_files):

        data_to_process = buffer['source_paths']
        no_folders = len(buffer['saving_paths'])

        for id_x in range(no_folders):
            full_saving_path = buffer['saving_paths'][id_x]
            label = buffer['labels'][id_x]
            matrix_merge = None

            for single_file_path in data_to_process[id_x]:
                loaded_file = np.load(single_file_path)
                preprocessed_data = self.possible_preprocessing[preprocessing_type](self.original_seq, loaded_file)

                if merge_files:
                    if matrix_merge is None:
                        matrix_merge = preprocessed_data
                    else:
                        matrix_merge = np.concatenate((matrix_merge, preprocessed_data), axis=0)

                else:
                    file_path = os.path.join(full_saving_path, f"file_{self.counters[data_type]}_label={label}.pt")
                    single_processed_data_torch = torch.Tensor(preprocessed_data)
                    torch.save(single_processed_data_torch, file_path)
                    self.counters[data_type] += 1

            if merge_files:
                file_path = os.path.join(full_saving_path, f"file_{self.counters[data_type]}_label={label}.pt")
                single_processed_data_torch = torch.Tensor(matrix_merge)
                torch.save(single_processed_data_torch, file_path)
                self.counters[data_type] += 1


    def get_saving_path(self,
                        mix_bool: bool,
                        preprocessing_type: str,
                        additional_folder_label: str,
                        data_type: str,
                        measurement_folder: str):
        if mix_bool:
            path_data_folder = os.path.join(self.data_cache_path,
                                            f"{preprocessing_type}{additional_folder_label}", data_type)
        else:
            path_data_folder = os.path.join(self.data_cache_path, f"{preprocessing_type}{additional_folder_label}",
                                            data_type,
                                            measurement_folder)

        return path_data_folder

    def preprocess_train_paths(self,
                               all_paths: list,
                               get_saving_path_train: Callable,
                               get_saving_path_valid: Callable,
                               split_train_into_train_and_valid,
                               split_ratio: float = 0.2):
        train_paths = []
        valid_paths = []

        for single_path in all_paths:
            saving_data_path = get_saving_path_train('')
            self.prepare_saving_path(saving_data_path)
            file_names = os.listdir(single_path)
            train_paths += self.prepare_full_paths(single_path, file_names)
            measurement_folder = Path(single_path).parts[-1]

            split_ratio = int(split_ratio * len(file_names))

            if split_train_into_train_and_valid:
                all_paths_full = np.asarray(copy(train_paths))
                np.random.shuffle(all_paths_full)
                train_paths = all_paths_full[split_ratio:].tolist()
                valid_paths = all_paths_full[:split_ratio].tolist()

                valid_saving_path = get_saving_path_valid(measurement_folder)
                self.prepare_saving_path(valid_saving_path)

                self.buffers_valid['source_paths'].append(valid_paths)
                self.buffers_valid['saving_paths'].append(valid_saving_path)
                self.buffers_valid['labels'].append(0)

            self.buffers_train['source_paths'].append(train_paths)
            self.buffers_train['saving_paths'].append(get_saving_path_train(''))
            self.buffers_train['labels'].append(0)
            print('')

    def preprocess_test_and_valid(self,
                                  buffer: dict,
                                  list_path: str,
                                  get_saving_path: Callable):

        for single_path in list_path:
            measurement_folder = Path(single_path).parts[-1]
            label = self.get_label(measurement_folder)

            saving_data_path = get_saving_path(measurement_folder)
            self.prepare_saving_path(saving_data_path)

            all_files_in_folder = os.listdir(single_path)
            data_batch_single_path = self.prepare_full_paths(single_path, all_files_in_folder)

            buffer['source_paths'].append(data_batch_single_path)
            buffer['saving_paths'].append(get_saving_path(measurement_folder))
            buffer['labels'].append(label)


    def preprocess_data(self,
                        data_paths: dict,
                        preprocessing_type: str,
                        mix_valid: bool = True,
                        mix_test: bool = True,
                        rewrite_data: bool = False,
                        merge_files: bool = False,
                        split_train_into_train_and_valid: bool = False,
                        additional_folder_label: str = '') -> dict:
        """
               Preprocess data for training, validation, and testing.

               Args:
                   data_paths (dict): A dictionary containing paths for Train, Valid, and Test data.
                   preprocessing_type (str): The type of preprocessing to apply.
                   mix_valid (bool): Whether to mix validation data.
                   mix_test (bool): Whether to mix test data.
                   rewrite_data (bool): Whether to rewrite existing data.
                   merge_files (bool): Whether to merge all files into a single file.
                   split_train_into_train_and_valid (bool): Split train data into train and valid
                   additional_folder_label (str): An additional label for the folder.

               Returns:
                   dict: A dictionary with paths for Train, Valid, and Test data.
               """
        if self.data_cache_path == None:
            raise DataProcessorException("Data cache path is not set")

        if len(self.original_seq) == 0:
            raise DataProcessorException("Original sequence is not set")


        #iterate over different data types (Train, Valid, Test)
        get_saving_path_train = lambda measurement_folder: self.get_saving_path(mix_bool=True,
                                                                                preprocessing_type=preprocessing_type,
                                                                                additional_folder_label=additional_folder_label,
                                                                                data_type='Train',
                                                                                measurement_folder=measurement_folder)

        get_saving_path_valid = lambda measurement_folder: self.get_saving_path(mix_bool=mix_valid,
                                                                                preprocessing_type=preprocessing_type,
                                                                                additional_folder_label=additional_folder_label,
                                                                                data_type='Valid',
                                                                                measurement_folder=measurement_folder)

        get_saving_path_test = lambda measurement_folder: self.get_saving_path(mix_bool=mix_test,
                                                                                preprocessing_type=preprocessing_type,
                                                                                additional_folder_label=additional_folder_label,
                                                                                data_type='Test',
                                                                                measurement_folder=measurement_folder)

        self.preprocess_train_paths(data_paths['Train'], get_saving_path_train, get_saving_path_valid, split_train_into_train_and_valid)
        self.preprocess_folder(data_type='Train',
                               buffer=self.buffers_train,
                               preprocessing_type=preprocessing_type,
                               merge_files=merge_files)

        self.preprocess_test_and_valid(self.buffers_valid, data_paths['Valid'], get_saving_path_valid)
        self.preprocess_folder(data_type='Valid',
                               buffer=self.buffers_valid,
                               preprocessing_type=preprocessing_type,
                               merge_files=merge_files)

        self.preprocess_test_and_valid(self.buffers_test, data_paths['Test'], get_saving_path_test)
        self.preprocess_folder(data_type='Test',
                               buffer=self.buffers_test,
                               preprocessing_type=preprocessing_type,
                               merge_files=merge_files)
