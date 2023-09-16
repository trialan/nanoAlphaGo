import numpy as np
import torch
import torch.nn as nn

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BOARD_SIZE, PASS
from nanoAlphaGo.rl.utils import _index_to_move


class PolicyNN(nn.Module):
    def __init__(self, color):
        super(PolicyNN, self).__init__()

        self.color = color
        assert color in [1, -1]

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc_policy = nn.Linear(256, BOARD_SIZE * BOARD_SIZE + 1)

    def forward(self, board_tensors_batch):
        x = self.conv1(board_tensors_batch)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)

        policy_outputs = self.fc_policy(x)

        normalised_policy_outputs = torch.softmax(policy_outputs, 1)
        assert_is_probs(normalised_policy_outputs)

        masks = _legal_move_mask(board_tensors_batch, self.color)
        masked_policy_outputs = normalised_policy_outputs * masks

        assert_sum_is_less_than_or_equal_to_one(masked_policy_outputs)
        self.assert_is_valid_move(masked_policy_outputs, board_tensors_batch)

        return masked_policy_outputs

    def get_move_as_int_from_prob_dist(self, prob_dist, board_tensor):
        predicted = torch.argmax(prob_dist)
        move_as_int = predicted.item()
        if move_as_int == PASS:
            return PASS
        return move_as_int

    def assert_is_valid_move(self, masked_policy_outputs, board_tensors_batch):
        for m, b in zip(masked_policy_outputs, board_tensors_batch):
            move_as_int = self.get_move_as_int_from_prob_dist(m,b)
            board = GoBoard(initial_state_matrix=b[0].cpu().numpy())
            position = _index_to_move(move_as_int)
            assert board.is_valid_move(position, self.color)


def assert_sum_is_less_than_or_equal_to_one(masked_policy_outputs):
    for x in masked_policy_outputs:
        if x.sum().item() < 1:
            return
        else:
            assert np.isclose(x.sum().item(), 1.0)


def _legal_move_mask(board_tensors_batch, player_color):
    masks = []
    for board_tensor in board_tensors_batch:
        matrix = board_tensor[0].cpu().numpy()
        board = GoBoard(initial_state_matrix=matrix)
        mask_size = BOARD_SIZE * BOARD_SIZE + 1
        mask = torch.zeros(mask_size, dtype=torch.float32)
        assert board._matrix.shape == (BOARD_SIZE, BOARD_SIZE)
        possible_moves = board.legal_moves(player_color)
        for move in possible_moves:
            if move == PASS:
                mask[-1] = 1
            else:
                x, y = move
                index = x * BOARD_SIZE + y
                mask[index] = 1
        masks.append(mask)
    return torch.stack(masks)


def assert_is_probs(x):
    assert np.isclose(x.sum().item(), 1.0)


if __name__ == '__main__':
    from nanoAlphaGo.config import WHITE, BOARD_SIZE
    board = GoBoard()
    model = PolicyNN(color=WHITE)

    prob_dist = model.forward(board.tensor)
    move = model.get_move_from_prob_dist(prob_dist)
    print("Predicted move:", move)


