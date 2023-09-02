import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

'''
Heuristic_model just calculates the line between the mid hips and shoulder. Then
divides it into 5 parts and returns the 4 points in between. Those are the baseline
prediction for the spine.
'''
def heuristic_model(data_df):
    data_mid_hip_x = (data_df['HipLeft_x'] + data_df['HipRight_x']) / 2
    data_mid_hip_y = (data_df['HipLeft_y'] + data_df['HipRight_y']) / 2
    data_mid_hip_z = (data_df['HipLeft_z'] + data_df['HipRight_z']) / 2

    data_mid_shoulder_x = (data_df['ShoulderLeft_x'] + data_df['ShoulderRight_x']) / 2
    data_mid_shoulder_y = (data_df['ShoulderLeft_y'] + data_df['ShoulderRight_y']) / 2
    data_mid_shoulder_z = (data_df['ShoulderLeft_z'] + data_df['ShoulderRight_z']) / 2

    data_delta_x = (data_mid_shoulder_x - data_mid_hip_x) / 5
    data_delta_y = (data_mid_shoulder_y - data_mid_hip_y) / 5
    data_delta_z = (data_mid_shoulder_z - data_mid_hip_z) / 5

    data_spine_xs = []
    data_spine_ys = []
    data_spine_zs = []

    for i in range(1, 5):
        data_spine_xs.append(data_mid_hip_x + data_delta_x * i)
        data_spine_ys.append(data_mid_hip_y + data_delta_y * i)
        data_spine_zs.append(data_mid_hip_z + data_delta_z * i)

    return data_spine_xs, data_spine_ys, data_spine_zs



# Define your MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, dropout_prob):
        super(FlexibleMLP, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, lengths):
        data = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        #data = x
        out, _ = self.lstm(data)
        # add dropout
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #out = self.fc2(out)
        #out = torch.relu(out)
        out = self.fc(out[:, -1, :])  # Get output from the last time step
        return out