import torch

from nanoAlphaGo.config import BOARD_SIZE, PASS


def _nn_tensor_from_matrix(board_matrix):
    """ Conv2D wants a 4D input. """
    x = torch.tensor(board_matrix, dtype=torch.float32)
    x = x.unsqueeze(0)
    return x


def _index_to_move(index):
    """ Returns a unique set of coordinates for each integer. """
    if index == BOARD_SIZE * BOARD_SIZE:
        return PASS
    else:
        x, y = divmod(index, BOARD_SIZE)
        return (x, y)

