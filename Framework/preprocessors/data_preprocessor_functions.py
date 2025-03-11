import numpy as np


class DataPreprocessorFunctions:
    """
    A class containing static methods for various data preprocessing functions.
    """

    @staticmethod
    def to_log(values: np.array) -> np.array:
        """
        Convert values to their logarithmic scale.

        Args:
            values (np.array): The input array of values.

        Returns:
            np.array: The logarithmic scale of the input values.
        """
        return -20 * np.log10(values)

    @staticmethod
    def estimate_channels(signal, original_sequence):
        """
        Estimate the channels by dividing the signal by the original sequence.

        Args:
            signal (np.array): The input signal array.
            original_sequence (np.array): The original sequence array.

        Returns:
            np.array: The estimated channels.
        """
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
        """
        Convert a complex array to an array with real and imaginary parts separated.

        Args:
            array (np.array): The input complex array.

        Returns:
            np.array: The array with real and imaginary parts separated.
        """
        real = array.real
        imag = array.imag

        ret_arr = []

        for real_sl, imag_sl in zip(real, imag):
            new_rr = np.concatenate((real_sl.reshape((-1, 1)), imag_sl.reshape((-1, 1))), axis=1)
            ret_arr.append(new_rr)

        return np.array(ret_arr)

    @staticmethod
    def reshape_arr(arr, no_signals_in_arr=2):
        """
        Reshape an array by splitting it into multiple signals.

        Args:
            arr (np.array): The input array.
            no_signals_in_arr (int): The number of signals in the array.

        Returns:
            np.array: The reshaped array.
        """
        all_signals = []
        for x in range(arr.shape[1] // no_signals_in_arr):
            single_signal = arr[:, (x * no_signals_in_arr):(x + 1) * no_signals_in_arr]
            single_signal_res = np.reshape(single_signal, (-1, 1), order='F')
            all_signals.append(single_signal_res)

        return np.asarray(all_signals)

    @staticmethod
    def split_by_groups_of_n(array_to_mean, len_groups: int = 4):
        """
        Split an array into groups of n and reshape it.

        Args:
            array_to_mean (np.array): The input array.
            len_groups (int): The length of each group.

        Returns:
            np.array: The reshaped array.
        """
        reshaped_array = array_to_mean.reshape(72, array_to_mean.shape[1] // len_groups, len_groups)
        result = reshaped_array.transpose(1, 0, 2)
        return np.array(result)

    @staticmethod
    def mean_by_quaters_axis_2(array_to_mean):
        """
        Calculate the mean of an array by quarters along the second axis.

        Args:
            array_to_mean (np.array): The input array.

        Returns:
            np.array: The array with the mean calculated by quarters.
        """
        new_arr = []
        for x in range(array_to_mean.shape[1] // 4):
            new_arr.append(array_to_mean[:, (4 * x):(4 * (x + 1))])
        new_arr = np.mean(np.array(new_arr), axis=2)
        new_arr = new_arr.T

        return new_arr

    @staticmethod
    def mean_by_quaters(array_to_mean):
        """
        Calculate the mean of an array by quarters.

        Args:
            array_to_mean (np.array): The input array.

        Returns:
            np.array: The array with the mean calculated by quarters.
        """
        new_arr = []
        for x in range(array_to_mean.shape[1] // 4):
            new_arr.append(array_to_mean[:, (4 * x):(4 * (x + 1))])

        new_arr = np.mean(np.array(new_arr), axis=0)

        return new_arr