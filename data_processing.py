import numpy as np
import torch
import sys
import traceback
import os
import plotly
import plotly.express as px
def estimate_channels(signal: np.ndarray, original_sequence: np.ndarray) -> np.ndarray:
    """
    Estimate the number of channels in the signal and return the signal with the estimated number of channels.
    Args:
    signal (np.ndarray): Array of PSS/SSS sequences (for each carrier).
    original_sequence (np.ndarray): Original PSS/SSS sequence.

    Returns:
    np.ndarray: Data as a numpy array.
    """

    multiply = signal.shape[1]//4
    new_seq = np.tile(original_sequence, multiply)

    # Compensate for different sequence lengths (48 vs 50)
    if not signal.shape[1] % 4 == 0:
        new_seq = np.concatenate((new_seq, original_sequence[:, :2]), axis=1)

    return signal/new_seq


def process_data(data_path: str, original_sequence_path: str) -> np.ndarray:
    """
    Process data from the specified path and return it as a numpy array.

    Args:
    data_path (str): Path to the data.
    original_sequence_path (str): Path to the original sequence.

    Returns:
    np.ndarray: Data as a numpy array.
    """

    original_sequence = np.load(original_sequence_path)
    data = None
    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            raw = np.load(os.path.join(data_path, file))
            estimated = estimate_channels(raw, original_sequence[:, :4])
            # estimated = estimated.reshape((estimated.shape[0]*2, estimated.shape[1]//2))
            estimated = np.abs(estimated)
            estimated = -20*np.log10(estimated)

            data = np.hstack((data, estimated)) if data is not None else estimated


    if data is None:
        raise FileNotFoundError(f"No .npy files found in {data_path}")
        traceback.print_exc()
        sys.exit(3)
    else:
        return data




if __name__ == "__main__":
    data = process_data("Data_selection/comeretial/","Data_selection/original.npy")
    data = torch.tensor(data, dtype=torch.float32)
    # torch.save(data, "Data_selection/commercial_abs_v2.pt")
    print(data.shape)