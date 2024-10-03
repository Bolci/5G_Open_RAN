import numpy as np
import torch
import os
import matplotlib.pyplot as plt


path = "/home/bolci/Documents/Projekty/5G_OPEN_RAN/Anomaly_detection/5G_Open_RAN/Data/Data_prepared/abs_only/Train/comeretial"
all_files = os.listdir(path)

for single_file in all_files:
    file_path = os.path.join(path, single_file)
    loaded_file = torch.load(file_path).numpy()

    plt.figure()
    plt.plot(loaded_file)
    plt.show()
    break