import numpy as np
import torch
import torch.nn as nn

from nanoAlphaGo.config import BOARD_SIZE, PASS
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.rl.masking import legal_move_mask
from nanoAlphaGo.rl.debugging import assert_no_nan_outputs
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
        self.set_device()

    def forward(self, board_tensors):
        board_tensors = board_tensors.to(self.device)
        outputs = self.forward_through_layers(board_tensors)
        masked_outputs = self.mask(board_tensors, outputs)
        normalised_outputs = self.normalise(masked_outputs)
        self.perform_sanity_checks(normalised_outputs, board_tensors)
        return normalised_outputs

    def set_device(self):
        self.device = torch.device("cpu")
        for layer in [self.conv1, self.conv2, self.fc1, self.fc_policy]:
            layer = layer.to(self.device)

    def forward_through_layers(self, board_tensors):
        x = self.conv1(board_tensors)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        policy_outputs = self.fc_policy(x)
        assert_no_nan_outputs(policy_outputs)
        return policy_outputs

    def normalise(self, policy_outputs):
        nonzero_mask = policy_outputs != 0
        policy_outputs[nonzero_mask] = torch.softmax(policy_outputs[nonzero_mask], dim=0)
        normalised_policy_outputs = policy_outputs
        assert_are_probs(normalised_policy_outputs)
        return normalised_policy_outputs

    def mask(self, board_tensors, outputs):
        """ Mask illegal moves. """
        masks = legal_move_mask(board_tensors, self.color)
        masked_outputs = outputs * masks
        return masked_outputs

    def perform_sanity_checks(self, outputs, board_tensors):
        assert_sum_is_less_than_or_equal_to_one(outputs)
        self.assert_is_valid_move(outputs, board_tensors)

    def get_move_as_int_from_prob_dist(self, prob_dist, board_tensor):
        predicted = torch.argmax(prob_dist)
        move_as_int = predicted.item()
        if move_as_int == PASS:
            return PASS
        return move_as_int

    def assert_is_valid_move(self, masked_policy_outputs, board_tensors):
        for m, b in zip(masked_policy_outputs, board_tensors):
            move_as_int = self.get_move_as_int_from_prob_dist(m,b)
            board = GoBoard(initial_state_matrix=b[0].cpu().numpy())
            position = _index_to_move(move_as_int)
            try:
                assert board.is_valid_move(position, self.color)
            except:
                import pdb;pdb.set_trace() 



def assert_sum_is_less_than_or_equal_to_one(masked_policy_outputs):
    sums = [x.sum().item() for x in masked_policy_outputs]
    for s in sums: assert_sum_is_leq_one(s)


def assert_sum_is_leq_one(s):
    if s < 1:
        return
    else:
        assert np.isclose(s, 1.0)


def assert_are_probs(x):
    for e in x: assert np.isclose(e.sum().item(), 1.0)


