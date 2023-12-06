import torch
import torch.nn as nn

from nanoAlphaGo.config import BOARD_SIZE, SEED
from nanoAlphaGo.rl.utils import set_seed


class ValueNN(nn.Module):
    def __init__(self):
        set_seed(SEED)
        super(ValueNN, self).__init__()
        board_size = BOARD_SIZE
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc_value = nn.Linear(256, 1)
        self.set_device()

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for layer in [self.conv1, self.conv2, self.fc1, self.fc_value]:
            layer = layer.to(self.device)

    def forward(self, board_tensors_batch):
        x = nn.functional.relu(self.conv1(board_tensors_batch))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        value_output = torch.tanh(self.fc_value(x))  # Scaled to [-1, 1]
        return value_output

