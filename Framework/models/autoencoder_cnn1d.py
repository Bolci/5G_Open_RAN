import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder1D(nn.Module):
    def __init__(self, input_length=72, in_channels=48, num_layers=4, base_channels=128, kernel=4, dropout=0.1):
        super(Autoencoder1D, self).__init__()
        self.name = "cnn1d_ae"
        self.num_layers = num_layers
        self.input_length = input_length
        self.in_channels = in_channels

        # Encoder
        encoder_layers = []
        # in_channels = 1
        out_channels = base_channels
        current_length = input_length
        self.intermediate_lengths = [input_length]

        for i in range(num_layers):
            kernel_size = kernel  # Fixed kernel size
            stride = 2  # Downsampling by 2
            padding = (kernel_size - 1) // 2  # Keep dimensions even

            encoder_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=dropout))
            in_channels = out_channels
            out_channels *= 2
            current_length = ((current_length + 2 * padding - (kernel_size - 1) - 1) // stride)+1
            self.intermediate_lengths.append(current_length)

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoded_length = current_length

        # Decoder
        decoder_layers = []
        out_channels = in_channels // 2

        for i in range(num_layers - 1, -1, -1):
            kernel_size = kernel  # Fixed kernel size
            stride = 2  # Upsampling by 2
            padding = (kernel_size - 1) // 2
            expected_output_size = self.intermediate_lengths[i]
            actual_output_size = (current_length - 1) * stride + kernel_size - 2 * padding
            output_padding = (expected_output_size - actual_output_size) if (expected_output_size - actual_output_size) < stride else 0

            decoder_layers.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(p=dropout))
            in_channels = out_channels
            out_channels //= 2
            current_length = expected_output_size

        decoder_layers.append(nn.Conv1d(in_channels, self.in_channels, kernel_size=3, padding=1))  # Final reconstruction layer
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # x - [batch, 50, 72]
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x


model = Autoencoder1D()
x = torch.randn(64, 48, 72)
model(x)