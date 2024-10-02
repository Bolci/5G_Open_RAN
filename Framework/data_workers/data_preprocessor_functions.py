import numpy as np


class DataPreprocessorFunctions:

    @staticmethod
    def to_log(values: np.array) -> np.array:
        return -20 * np.log10(values)

    @staticmethod
    def estimate_channels(signal, original_sequence):
        new_seq = np.array([])
        multiply = signal.shape[1] // 4

        for x in range(multiply):
            if len(new_seq) == 0:
                new_seq = original_sequence
            else:
                new_seq = np.concatenate((new_seq, original_sequence), axis=1)

        if not signal.shape[1] % 4 == 0:
            new_seq = np.concatenate((new_seq, original_sequence[:, :2]), axis=1)

        return signal / new_seq

    @staticmethod
    def complex_to_arr(array):
        real = array.real
        imag = array.imag

        ret_arr = []

        for real_sl, imag_sl in zip(real, imag):
            new_rr = np.concatenate((real_sl.reshape((-1, 1)), imag_sl.reshape((-1, 1))), axis=1)
            ret_arr.append(new_rr)

        return np.array(ret_arr)

    @staticmethod
    def reshape_arr(arr, no_signals_in_arr=2):
        all_signals = []
        for x in range(arr.shape[1] // no_signals_in_arr):
            single_signal = arr[:, (x * no_signals_in_arr):(x + 1) * no_signals_in_arr]

            # single_signal_rs2 = np.concatenate((single_signal[:,0], single_signal[:,1]), axis=0).reshape((-1,1))
            single_signal_res = np.reshape(single_signal, (-1, 1), order='F')
            all_signals.append(single_signal_res)

        return np.asarray(all_signals)