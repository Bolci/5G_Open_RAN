import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

#path_to_data = '/home/bolci/Documents/Projekty/5G_OPEN_RAN/Anomaly_detection/5G_Open_RAN/Data/Data_prepared/abs_only/Train/comeretial'
path_to_data = '/home/bolci/Documents/Projekty/5G_OPEN_RAN/Anomaly_detection/5G_Open_RAN/Data/Data_prepared/abs_only/Valid'
all_files = os.listdir(path_to_data)

plt.figure()
counter = 0

for single_file in all_files:
    single_file_path = os.path.join(path_to_data, single_file)
    single_tensor = torch.load(single_file_path).numpy()
    plt.plot(single_tensor)

    if counter > 10:
        break

    counter += 1

plt.show()




'''
# Define the PyTorch model class
class ConvNet(nn.Module):
    def __init__(self, conv_1_filter, conv_1_kernel, conv_2_filter, conv_2_kernel, conv_3_filter, conv_3_kernel,
                 dense_1_units, dense_2_units, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_1_filter, kernel_size=conv_1_kernel, activation=F.silu)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(conv_1_filter, conv_2_filter, kernel_size=conv_2_kernel, activation=F.silu)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(conv_2_filter, conv_3_filter, kernel_size=conv_3_kernel, activation=F.silu)

        # Calculate the size of the flattened output from the conv layers
        self.flatten_size = self._get_conv_output_size()

        self.fc1 = nn.Linear(self.flatten_size, dense_1_units)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(dense_1_units, dense_2_units)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(dense_2_units, num_classes)

    def _get_conv_output_size(self):
        # Dummy input to compute the flattened output size after convolutional layers
        dummy_input = torch.zeros(1, 1, 45, 51)
        x = self.pool1(self.conv1(dummy_input))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x):
        x = self.pool1(F.silu(self.conv1(x)))
        x = self.pool2(F.silu(self.conv2(x)))
        x = F.silu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout1(F.silu(self.fc1(x)))
        x = self.dropout2(F.silu(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# Objective function for hyperparameter tuning with Optuna
def objective(trial):
    # Define hyperparameters
    conv_1_filter = trial.suggest_int('conv_1_filter', 8, 128, step=16)
    conv_1_kernel = trial.suggest_categorical('conv_1_kernel', [3, 5])

    conv_2_filter = trial.suggest_int('conv_2_filter', 8, 64, step=16)
    conv_2_kernel = trial.suggest_categorical('conv_2_kernel', [3, 5])

    conv_3_filter = trial.suggest_int('conv_3_filter', 8, 64, step=16)
    conv_3_kernel = trial.suggest_categorical('conv_3_kernel', [3, 5])

    dense_1_units = trial.suggest_int('dense_1_units', 32, 128, step=16)
    dense_2_units = trial.suggest_int('dense_2_units', 32, 128, step=16)

    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 5e-4])

    # Instantiate the model
    model = ConvNet(conv_1_filter, conv_1_kernel, conv_2_filter, conv_2_kernel, conv_3_filter, conv_3_kernel,
                    dense_1_units, dense_2_units, num_classes=10)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Dummy training loop (replace with actual dataloaders)
    for epoch in range(10):
        # Forward pass, loss calculation, backward pass, and optimizer step
        optimizer.zero_grad()
        # Assume `inputs` and `labels` are your training data batches
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation accuracy as objective metric (replace with actual validation)
    val_accuracy = evaluate(model, val_loader)  # Placeholder function
    return val_accuracy

# Running the Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Output summary of the search space
print(study.best_params)
'''
