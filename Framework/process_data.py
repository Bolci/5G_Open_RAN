# This script is used to process the data and create torch tensors for individual datasets - channel responses in a complex form
import numpy as np
import torch
import os
import argparse


def estimate_channels(
    signal: np.ndarray, original_sequence: np.ndarray
) -> np.ndarray:
    """
    Estimate the number of channels in the signal and return the signal with the estimated number of channels.
    Args:
    signal (np.ndarray): Array of PSS/SSS sequences (for each carrier).
    original_sequence (np.ndarray): Original PSS/SSS sequence.

    Returns:
    np.ndarray: Data as a numpy array.
    """
    multiply = signal.shape[1] // 4
    new_seq = np.tile(original_sequence, multiply)

    # Compensate for different sequence lengths (48 vs 50)
    if not signal.shape[1] % 4 == 0:
        new_seq = np.concatenate((new_seq, original_sequence[:, :2]), axis=1)

    return signal / new_seq


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

    # Compute the average of the original PSS/SSS sequence (repeats every 4 samples)
    original_sequence = original_sequence.reshape(
        (original_sequence.shape[0], original_sequence.shape[1] // 4, 4)
    )
    original_sequence = np.mean(original_sequence, axis=2)

    data = None
    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            raw = np.load(os.path.join(data_path, file))
            estimated = estimate_channels(raw, original_sequence)

            data = (
                np.hstack((data, estimated)) if data is not None else estimated
            )

    if data is None:
        raise FileNotFoundError(f"No .npy files found in {data_path}")
    else:
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data and create torch tensors for individual datasets - channel responses in a complex form."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="C:/Users/xzelen23/PycharmProjects/5G_Open_RAN/Data_selection/Fake_Bts_PCI_466_wPA_traffic",
    )
    parser.add_argument(
        "--original_sequence_path",
        type=str,
        default="C:/Users/xzelen23/PycharmProjects/5G_Open_RAN/Data_selection/original.npy",
    )
    # parser.add_argument("--save_path", type=str, default="Data_selection/commercial/")
    # parser.add_argument("--save_name", type=str, default="channel_responses.pt")
    args = parser.parse_args()
    print(
        f"Processing the data from {args.data_path} and saving it to {args.data_path} as channel_responses.pt"
    )

    data = process_data(args.data_path, args.original_sequence_path)
    data = torch.tensor(data, dtype=torch.complex64)
    torch.save(data, os.path.join(args.data_path, "channel_responses.pt"))
    print("Data processed and saved successfully.")
