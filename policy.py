import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nanoAlphaGo.board import GoBoard


class PolicyNN(nn.Module):
    def __init__(self, board, color):
        super(PolicyNN, self).__init__()
        board_size = board.size

        self.color = color
        assert color in [1, -1]

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, board_size * board_size + 1)

    def generate_move(self, board):
        output = self._forward_pass(board)
        mask = _legal_move_mask(board, self.color)
        output = output * mask
        _, predicted = torch.max(output, 1)
        move_as_int = predicted.item()
        move_as_coordinates = _index_to_move(move_as_int, board.size)
        return move_as_coordinates

    def _forward_pass(self, board):
        x = _format_board_for_nn(board)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _legal_move_mask(board, color):
    board_size = board.size
    mask_size = board_size * board_size + 1
    mask = torch.zeros(mask_size, dtype=torch.float32)
    possible_moves = board.legal_moves(color)
    for move in possible_moves:
        if move == 'pass':
            mask[-1] = 1
        else:
            x, y = move
            index = x * board_size + y
            mask[index] = 1
    return mask


def _format_board_for_nn(board):
    """ Conv2D wants a 4D input. """
    x = torch.tensor(board.board, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0)
    return x


def _index_to_move(index, board_size):
    """ Returns a unique set of coordinates for each integer. """
    if index == board_size * board_size:
        return "pass"
    else:
        x, y = divmod(index, board_size)
        return (x, y)


if __name__ == '__main__':
    board = GoBoard(size=19)
    model = PolicyNN(board, color=1)
    move = model.generate_move(board)
    print("Predicted move:", move)


