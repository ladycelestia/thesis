import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,dropout,activation_fn,activation_fn1, norm_layer_type):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = dropout if num_layers > 1 else 0   

        #self.lstm = LSTMCustom(input_size, hidden_size, lstm_dropout, num_layers, activation_fn, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)

        # Additional fully connected layers with batch normalization
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        if norm_layer_type == 'batch_norm':
            self.norm_layer_type = nn.BatchNorm1d(hidden_size)
        elif norm_layer_type == 'layer_norm':
            self.norm_layer_type = nn.LayerNorm(hidden_size)
        else:
            self.norm_layer_type = nn.Identity()
        

        self.fc_out = nn.Linear(hidden_size, output_size)
    
        self.activation_fn = activation_fn #tanh or relu
        self.activation_fn1 = activation_fn1 #sigmoid or identity
    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get last time step's output

        # Pass through fully connected layers with activation and batch normalization
        out = self.fc1(out)
        out = self.norm_layer_type(out)
        out = self.activation_fn(out)  # Apply chosen activation function
        out = self.dropout(out)  # Apply dropout after activation
        
        
        # Final output layer with sigmoid
        out = self.fc_out(out)
        out = self.activation_fn1(out)

        return out