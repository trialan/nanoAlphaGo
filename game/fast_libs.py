import ctypes
import numpy as np
from nanoAlphaGo.config import BOARD_SIZE

class Position(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int),
                ("y", ctypes.c_int)]


path = "/Users/thomasrialan/Documents/code/nanoAlphaGo/game/liberties.so"
liberties_lib = ctypes.CDLL(path)
liberties_lib.count_liberties.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.c_int, Position]
liberties_lib.count_liberties.restype = ctypes.c_int


def fast_count_liberties(coords, matrix):
    x, y = coords
    position = Position(x, y)

    matrix_size = BOARD_SIZE
    matrix_data = [int(v) for v in matrix.flatten()]

    # Create an array of ctypes.c_int arrays, each representing a row
    RowArrayType = ctypes.c_int * matrix_size
    c_matrix_rows = (RowArrayType * matrix_size)()

    # Populate the ctypes array with data from the numpy matrix
    for i in range(matrix_size):
        row = matrix_data[i * matrix_size : (i + 1) * matrix_size]
        c_matrix_rows[i] = RowArrayType(*row)

    # Cast the row arrays to ctypes.POINTER(ctypes.c_int)
    c_matrix_ptrs = (ctypes.POINTER(ctypes.c_int) * matrix_size)()
    for i in range(matrix_size):
        c_matrix_ptrs[i] = ctypes.cast(c_matrix_rows[i], ctypes.POINTER(ctypes.c_int))

    n_liberties = liberties_lib.count_liberties(c_matrix_ptrs, matrix_size, position)
    return n_liberties


