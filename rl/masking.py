import torch
import numpy as np

from nanoAlphaGo.config import BOARD_SIZE, PASS
from nanoAlphaGo.game.board import GoBoard


def legal_move_mask(board_tensors, player_color):
    masks = [generate_mask(board_tensor, player_color) for board_tensor in board_tensors]
    masks = torch.stack(masks).to(board_tensors.device)
    return masks


def generate_mask(board_tensor, player_color):
    matrix = board_tensor[0].cpu().numpy()
    board = GoBoard(initial_state_matrix=matrix)
    assert board._matrix.shape == (BOARD_SIZE, BOARD_SIZE)
    mask = torch.zeros(BOARD_SIZE * BOARD_SIZE + 1, dtype=torch.float32)
    possible_moves = board.legal_moves(player_color)
    return _populate_mask_with_moves(mask, possible_moves)


def _populate_mask_with_moves(mask, possible_moves):
    moves = [move for move in possible_moves if move != PASS]
    if moves:
        indices = np.array([(x * BOARD_SIZE + y) for x, y in moves])
        mask[indices] = 1
    if PASS in possible_moves:
        mask[-1] = 1
    return mask


