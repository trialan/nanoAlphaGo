import numpy as np
import torch
import torch.nn as nn

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BOARD_SIZE, PASS
from nanoAlphaGo.rl.utils import _format_board_for_nn, _index_to_move


class PolicyNN(nn.Module):
    def __init__(self, color):
        super(PolicyNN, self).__init__()

        self.color = color
        assert color in [1, -1]

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc_policy = nn.Linear(256, BOARD_SIZE * BOARD_SIZE + 1)

    def forward(self, board):
        x = _format_board_for_nn(board)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))

        policy_output = self.fc_policy(x)
        normalised_policy_output = torch.sigmoid(policy_output)
        return normalised_policy_output

    def get_policy_output(self, board):
        raw_policy_output = self.forward(board)
        mask = _legal_move_mask(board, self.color)
        policy_output = raw_policy_output * mask
        _, predicted = torch.max(policy_output, 1)
        move_as_int = predicted.item()
        return move_as_int, policy_output

    def get_move(self, board):
        move_as_int, probs = self.get_policy_output(board)
        move_as_coordinates = _index_to_move(move_as_int)
        assert board.is_valid_move(move_as_coordinates, self.color)
        return move_as_coordinates



def _legal_move_mask(board, color):
    mask_size = BOARD_SIZE * BOARD_SIZE + 1
    mask = torch.zeros(mask_size, dtype=torch.float32)
    possible_moves = board.legal_moves(color)
    for move in possible_moves:
        if move == PASS:
            mask[-1] = 1
        else:
            x, y = move
            index = x * BOARD_SIZE + y
            mask[index] = 1
    return mask



if __name__ == '__main__':
    from nanoAlphaGo.config import WHITE, BOARD_SIZE
    board = GoBoard()
    model = PolicyNN(color=WHITE)
    move, _ = model.get_policy_output(board)
    print("Predicted move:", move)


