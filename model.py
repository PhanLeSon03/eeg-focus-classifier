import torch
import torch.nn as nn
import torch.nn.functional as F

class MC_Model_1D(nn.Module):
    def __init__(self, n_channels=2, n_features = 18, time_steps=64, num_classes=2):
        super(MC_Model_1D, self).__init__()

        self.n_channels = n_channels
        self.time_steps = time_steps

        self.dense_input = nn.Linear(n_channels*n_features, 32)  #  original input_size is 46
        self.relu = nn.ELU()
        self.drop_dense = nn.Dropout(0.4)

        self.gru1 = nn.GRU(input_size=32, hidden_size=8, batch_first=True, bidirectional=True)
        self.drop_gru1 = nn.Dropout(0.4)

        self.gru2 = nn.GRU(input_size=16, hidden_size=4, batch_first=True, bidirectional=True)
        self.drop_gru2 = nn.Dropout(0.4)

        self.output = nn.Linear(time_steps*8, num_classes)

    def forward(self, x):
        # x shape: (B, T, F=46)
        x = self.relu(self.dense_input(x)) 
        x = self.drop_dense(x)
        
        x, _ = self.gru1(x)
        x = self.drop_gru1(x)

        x, _ = self.gru2(x)
        x = self.drop_gru2(x)

        # x = x[:, -1, :]  # Take last time step
        x = x.reshape(x.size(0), -1) # (B, 64*64)
        return torch.softmax(self.output(x), dim=1)

    def regularization_loss(self, lambda_ortho=1e-1, gamma_sparse=1e-1):
        return 0  # Define if needed

