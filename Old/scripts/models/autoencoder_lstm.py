import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super(LSTMAutoencoder, self).__init__()
        

        self.encoder_lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)

        self.latent_lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)

        self.decoder_lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        self.output_layer = nn.Linear(64, 2)
    
    def forward(self, x):

        encoded, _ = self.encoder_lstm(x)
        latent, _ = self.latent_lstm(encoded)
        decoded, _ = self.decoder_lstm(latent)
        output = self.output_layer(decoded)
        
        return output
        