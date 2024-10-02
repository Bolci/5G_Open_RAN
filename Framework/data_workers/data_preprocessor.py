import numpy as np

from .data_preprocessor_functions import DataPreprocessorFunctions
import os
from ..exceptions.exceptions import DataProcessorException
from ..utils.utils import load_mat_file

class PreprocessorTypes:
    @staticmethod
    def abs_only(self, raw_data):
        pass

    @staticmethod
    def mean_by_quaters(array_to_mean):
        new_arr = []
        for x in (range(array_to_mean.shape[1] // 4)):
            new_arr.append(array_to_mean[:, 4 * x:(4 * (x + 1))])

        new_arr = np.mean(np.array(new_arr), axis=0)

        return new_arr


class DataPreprocessor(PreprocessorTypes):
    def __init__(self):
        super().__init__()
        self.data_cache_path = None
        self.original_seq = []
        self.possible_preprocessing = {1: ""}


    def set_cache_path(self, path: str) -> None:
        self.data_cache_path = path

    def set_original_seg(self, path: str) -> None:
        mat_file = load_mat_file(path)
        pss_sss_raw = mat_file['rxGridSSBurst']
        pss_sss_raw = PreprocessorTypes.mean_by_quaters(pss_sss_raw)
        self.original_seq = pss_sss_raw

    def prepare_saving_path(self, saving_folder_name: str) -> str:
        full_path = os.path.join(self.data_cache_path, saving_folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        return full_path

    def preprocess_data(self, preprocessing_type: str) -> str:
        if self.data_cache_path == None:
            raise DataProcessorException("Data cache path is not set")

        if len(self.original_seq) == 0:
            raise DataProcessorException("Original sequence is not set")

        full_data_path = self.prepare_saving_path('abs_only')

        return full_data_path
