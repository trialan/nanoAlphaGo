import torch
import torch.nn as nn

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.rl.utils import _format_board_for_nn


def value_function(board_states, valueNN):
    values = []
    board = GoBoard()
    for state in board_states:
        board.matrix = state
        value = valueNN.get_value(board)
        values.append(value)
    return values


class ValueNN(nn.Module):
    def __init__(self, board):
        super(ValueNN, self).__init__()
        board_size = board.size

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc_value = nn.Linear(256, 1)  # Value head

    def forward(self, board):
        x = _format_board_for_nn(board)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))

        value_output = torch.tanh(self.fc_value(x))  # Scaled to [-1, 1]
        return value_output

    def get_value(self, board):
        value_output = self.forward(board)
        return value_output.item()

