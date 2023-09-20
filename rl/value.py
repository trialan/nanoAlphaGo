import torch
import torch.nn as nn

from nanoAlphaGo.game.board import GoBoard


class ValueNN(nn.Module):
    def __init__(self, board):
        super(ValueNN, self).__init__()
        board_size = board.size

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc_value = nn.Linear(256, 1)  # Value head

    def forward(self, board_tensors_batch):
        x = nn.functional.relu(self.conv1(board_tensors_batch))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))

        value_output = torch.tanh(self.fc_value(x))  # Scaled to [-1, 1]
        return value_output

    def get_value(self, board):
        value_output = self.forward(board)
        return value_output.item()


