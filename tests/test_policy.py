import numpy as np
import pytest
import torch

from nanoAlphaGo.config import PASS, WHITE, BLACK, BOARD_SIZE
from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.rl.policy import _index_to_move, PolicyNN, legal_move_mask
from nanoAlphaGo.rl.utils import _nn_tensor_from_matrix


def test_that_model_is_outputting_moves_as_coordinates():
    board = GoBoard()
    policy = PolicyNN(color=1)
    for _ in range(100):
        #Must unsqueeze as forward expects batches
        batch = board.tensor.unsqueeze(0)
        move_probs = policy.forward(batch)
        move = policy.get_move_as_int_from_prob_dist(move_probs, board.tensor)
        move = _index_to_move(move)
        assert_move_is_a_pair_of_coordinates_or_pass(move)


def test_that_index_to_move_mapping_is_unique():
    board = GoBoard()
    coordinates = []
    for i in range(board.size**2+1):
        move = _index_to_move(i)
        coordinates.append(move)
    assert len(set(coordinates)) == board.size**2 + 1


def test_we_are_masking_the_right_number_of_moves():
    board = GoBoard()
    batch = board.tensor.unsqueeze(0)
    # Function expects a batch of tensors
    masks = legal_move_mask(batch, BLACK)
    assert len(masks) == 1
    assert torch.sum(masks) == BOARD_SIZE * BOARD_SIZE + 1

    board = _setup_a_complicated_board()
    batch = board.tensor.unsqueeze(0)
    masks = legal_move_mask(batch, WHITE)
    number_of_stones_on_board = np.count_nonzero(board._matrix)
    assert torch.sum(masks) == BOARD_SIZE * BOARD_SIZE + 1 - number_of_stones_on_board


def test_that_sanity_checks_are_sane():
    """ Policy isn't allowed to output an illegal move
        thanks to our masking. """
    board = _setup_a_complicated_board()
    batch = board.tensor.unsqueeze(0)
    policy = PolicyNN(color=1, sampler="argmax")
    #n.b. this wouldn't pass with multinomial sampling by construction
    #because we make the most likely move a bad move, and wait to see
    #that this move isn't allowed.
    move_probs = policy.forward(batch)
    move_probs[0][0] = 0.5

    with pytest.raises(AssertionError):
        policy.assert_is_valid_move(move_probs, batch)

    move_probs[0][0] = 0.
    move_probs[0][1] = 0.5

    with pytest.raises(AssertionError):
        policy.assert_is_valid_move(move_probs, batch)

    move_probs[0][1] = 0.
    move_probs[0][3] = 0.5
    policy.assert_is_valid_move(move_probs, batch)


def _setup_a_complicated_board():
    matrix = np.array([[1,1,-1,0,0,0,0,0,0],
                        [1,0,-1,0,0,0,0,0,0],
                        [-1,-1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,-1,-1,0,0,-1,0,0],
                        [0,0,-1, 1,-1,0,0,0,0],
                        [0,-1,1,1,1,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0]])
    board = GoBoard(size=9, initial_state_matrix=matrix)
    return board


def assert_move_is_a_pair_of_coordinates_or_pass(move):
    if move == PASS:
        assert True
        return
    assert type(move) == tuple
    assert len(move) == 2
    assert type(move[0]) == int
    assert type(move[1]) == int
