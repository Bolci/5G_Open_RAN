import numpy as np
import torch
from .data_preprocessor_functions import DataPreprocessorFunctions
import os
from ..exceptions.exceptions import DataProcessorException
from ..utils.utils import load_mat_file
from pathlib import Path


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
        """
        Initialize the DataPreprocessor with default values.
        """
        super().__init__()
        self.data_cache_path = None
        self.original_seq = []
        self.possible_preprocessing = {
            'abs_only': lambda x, y: PreprocessorTypes.abs_only(x, y),
            'abs_only_mean_by_group': lambda x, y: PreprocessorTypes.abs_only_mean_by_group(x, y),
            'abs_only_by_one_sample': lambda x, y: PreprocessorTypes.abs_only_by_one(x, y),
            'abs_only_multichannel': lambda x, y: PreprocessorTypes.abs_only_multichannel(x, y),
            'raw_IQ': lambda x, y: PreprocessorTypes.raw_IQ(x, y),
        }
        self.counters = {"Train": 0, "Valid": 0, "Test": 0}
        self.paths_for_datasets = {'Train': [], 'Test': [], 'Valid': []}

        self.train_paths_buffer = []
        self.valid_paths_buffer = []
        self.test_paths_buffer = []

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

    def prepare_saving_path(self, saving_folder_name: str) -> str:
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

        return full_path

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
                          data_type: str,
                          all_files_paths: str,
                          preprocessing_type: str,
                          label: int,
                          full_saving_path: str,
                          merge_files: bool = False) -> None:
        """
        Preprocess all files in a folder and save the preprocessed data.

        Args:
            data_type (str): The type of data (Train, Valid, Test).
            source_folder_path (str): The path to the source folder containing raw data files.
            preprocessing_type (str): The type of preprocessing to apply.
            label (int): The label for the data.
            full_saving_path (str): The path to save the preprocessed data.
            merge_files (bool): Whether to merge all files into a single file.
        """
        #all_files = os.listdir(source_folder_path)
        if merge_files:
            single_matrix = None
            for single_file_path in all_files_paths:
                #single_file_path = os.path.join(source_folder_path, single_file_name)
                loaded_file = np.load(single_file_path)
                preprocessed_data = self.possible_preprocessing[preprocessing_type](self.original_seq, loaded_file)

                if single_matrix is None:
                    single_matrix = preprocessed_data
                else:
                    single_matrix = np.concatenate((single_matrix, preprocessed_data), axis=0)

            file_path = os.path.join(full_saving_path, f"file_{self.counters[data_type]}_label={label}.pt")
            single_processed_data_torch = torch.Tensor(single_matrix)
            torch.save(single_processed_data_torch, file_path)
            self.counters[data_type] += 1

        else:
            for single_file_path in all_files_paths:
                loaded_file = np.load(single_file_path)
                preprocessed_data = self.possible_preprocessing[preprocessing_type](self.original_seq, loaded_file)


                for single_processed_data in preprocessed_data:
                    file_path = os.path.join(full_saving_path, f"file_{self.counters[data_type]}_label={label}.pt")
                    single_processed_data_torch = torch.Tensor(single_processed_data)
                    torch.save(single_processed_data_torch, file_path)

                    self.counters[data_type] += 1

    def preprocess_train_paths(self,
                               all_paths,
                               split_train_into_train_and_valid,
                               split_ratio: float = 0.2):
        all_paths_full = []
        train_paths = []
        valid_paths = []

        for single_path in all_paths:
            file_names = os.listdir(single_path)
            all_paths_full += self.prepare_full_paths(single_path, file_names)

        if split_train_into_train_and_valid:
            all_paths_full = np.asarray(all_paths_full)
            np.random.shuffle(all_paths_full)
            train_paths = all_paths_full[split_ratio:].tolist()
            valid_paths = all_paths_full[:split_ratio].tolist()
        else:
            train_paths = all_paths_full

        return train_paths, valid_paths


    def preprocess_data(self,
                        data_paths: dict,
                        preprocessing_type: str,
                        mix_valid: bool = True,
                        mix_test: bool = True,
                        rewrite_data: bool = False,
                        merge_files: bool = False,
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
            additional_folder_label (str): An additional label for the folder.

        Returns:
            dict: A dictionary with paths for Train, Valid, and Test data.
        """
        if self.data_cache_path is None:
            raise DataProcessorException("Data cache path is not set")

        if len(self.original_seq) == 0:
            raise DataProcessorException("Original sequence is not set")

        for data_type, list_path in data_paths.items():
            mix_bool = False
            if data_type == 'Valid' and mix_valid:
                mix_bool = True

            if data_type == 'Test' and mix_test:
                mix_bool = True

            if data_type == 'Train':
                train_paths, valid_paths = self.preprocess_train_paths(list_path)
                path_data_folder = os.path.join(self.data_cache_path,
                                                f"{preprocessing_type}{additional_folder_label}",
                                                data_type)
                full_data_path = self.prepare_saving_path(path_data_folder)
                self.paths_for_datasets[data_type].append(full_data_path)

                if rewrite_data:
                    self.preprocess_folder(data_type=data_type,
                                           all_files_paths=train_paths,
                                           preprocessing_type=preprocessing_type,
                                           label=0,
                                           full_saving_path=full_data_path,
                                           merge_files=merge_files)

                if not valid_paths == []:
                    if not mix_bool:
                        path_data_folder = os.path.join(self.data_cache_path,
                                                        f"{preprocessing_type}{additional_folder_label}",
                                                        'Valid',
                                                        'from_train')
                    else:
                        path_data_folder = os.path.join(self.data_cache_path,
                                                        f"{preprocessing_type}{additional_folder_label}",
                                                        'Valid')

                    full_data_path = self.prepare_saving_path(path_data_folder)

                    if full_data_path not in self.paths_for_datasets[data_type]:
                        self.paths_for_datasets[data_type].append(full_data_path)

                    if rewrite_data:
                        self.preprocess_folder(data_type='Valid',
                                               all_files_paths=valid_paths,
                                               preprocessing_type=preprocessing_type,
                                               label=0,
                                               full_saving_path=full_data_path,
                                               merge_files=merge_files)


            for single_path in list_path:
                measurement_folder = Path(single_path).parts[-1]
                label = self.get_label(measurement_folder)

                if not mix_bool:
                    path_data_folder = os.path.join(self.data_cache_path,
                                                    f"{preprocessing_type}{additional_folder_label}",
                                                    data_type,
                                                    measurement_folder)
                else:
                    path_data_folder = os.path.join(self.data_cache_path,
                                                    f"{preprocessing_type}{additional_folder_label}",
                                                    data_type)

                full_data_path = self.prepare_saving_path(path_data_folder)

                if full_data_path not in self.paths_for_datasets[data_type]:
                    self.paths_for_datasets[data_type].append(full_data_path)

                if rewrite_data:
                    all_files_in_dir = os.listdir(single_path)
                    all_files_path = self.prepare_full_paths(single_path, all_files_in_dir)

                    self.preprocess_folder(data_type=data_type,
                                           all_files_paths=all_files_path,
                                           preprocessing_type=preprocessing_type,
                                           label=label,
                                           full_saving_path=full_data_path,
                                           merge_files=merge_files)

        return self.paths_for_datasets