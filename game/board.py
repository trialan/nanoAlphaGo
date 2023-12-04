import numpy as np

from nanoAlphaGo.config import BOARD_SIZE, PASS
from nanoAlphaGo.rl.utils import _index_to_move, _nn_tensor_from_matrix
from nanoAlphaGo.game.scoring import NEIGHBORS, position_is_within_board
from nanoAlphaGo.graphics.rendering import display_board

"""
Bad things about this code:
    - It's confusing to have both self._matrix and self.tensor,
      ideally we'd only have self.tensor.
      --> indeed this caused a bug (now fixed).

Checks:
    assert_board_is_self_consistent(board): matrix == tensor
"""

class GoBoard:
    def __init__(self, size=BOARD_SIZE, initial_state_matrix=None):
        self.size = size
        self.initialise_board(initial_state_matrix)
        self.previous_board = None  # To check for Ko rule
        #assert_board_is_self_consistent(self)

    def initialise_board(self, initial_state_matrix):
        if initial_state_matrix is None:
            self._matrix = np.zeros((self.size, self.size), dtype=int)
        else:
            self._matrix = initial_state_matrix
        self.tensor = _nn_tensor_from_matrix(self._matrix)

    def legal_moves(self, color):
        """ Doesn't return the PASS move """
        #assert_board_is_self_consistent(self)
        empty_positions = np.column_stack(np.where(self._matrix == 0))
        moves = [(x, y) for x, y in empty_positions if self.is_valid_move((x, y), color)]
        #assert_board_is_self_consistent(self)
        return moves

    def is_valid_move(self, position, color):
        #assert_board_is_self_consistent(self)
        if position == PASS:
            return True

        x, y = position
        valid = True

        # Out-of-bounds check
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            valid = False

        # Intersection check
        elif self._matrix[x, y] != 0:
            valid = False

        else:
            # Temporary place the stone to check for suicide rule
            self._matrix[x, y] = color
            self.tensor[0][x,y] = color

            if self.count_liberties(position) == 0:
                if self.neighbours_all_have_no_liberty((x,y)):
                    valid = True #it's not suicide if you capture
                else:
                    valid = False
            self._matrix[x, y] = 0  # Reset the position to its original state
            self.tensor[0][x,y] = 0

        # TODO: Check for Ko rule
        #assert_board_is_self_consistent(self)
        return valid

    def neighbours_all_have_no_liberty(self, position):
        x,y = position
        neighbors = [n for n in NEIGHBORS[position] if position_is_within_board(n)]
        neighbor_libs = [self.count_liberties(n) for n in neighbors]
        return sum(neighbor_libs) == 0

    def count_liberties(self, position):
        #assert_board_is_self_consistent(self)
        stack = [position]
        liberties_set = set()
        visited = set()

        color = self._matrix[position]

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                    continue

                if self._matrix[nx, ny] == 0:
                    liberties_set.add((nx, ny))

                elif self._matrix[nx, ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

        n_unique_liberties = len(liberties_set)
        #assert_board_is_self_consistent(self)
        return n_unique_liberties

    def apply_move(self, move, color):
        #assert_board_is_self_consistent(self)
        move = _index_to_move(move)
        assert self.is_valid_move(move, color)

        if move == PASS:
            return

        x, y = move
        self._matrix[x, y] = color
        self.tensor = _nn_tensor_from_matrix(self._matrix)

        opponent = -color
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                continue
            if self._matrix[nx, ny] == opponent:
                if self.count_liberties((nx, ny)) == 0:
                    captured_stones = self._remove_group((nx, ny))
        self.assert_position_is_occupied(x,y)
        #assert_board_is_self_consistent(self)

    def _remove_group(self, position):
        #assert_board_is_self_consistent(self)
        stack = [position]
        color = self._matrix[position]
        captured_count = 0

        while stack:
            x, y = stack.pop()

            if self._matrix[x, y] == color:
                self._matrix[x, y] = 0  # Remove the stone
                self.tensor = _nn_tensor_from_matrix(self._matrix)
                captured_count += 1  # Increment the number of captured stones

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy

                    if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                        continue

                    stack.append((nx, ny))
        #assert_board_is_self_consistent(self)
        return captured_count

    def assert_position_is_occupied(self, x, y):
        assert self._matrix[x,y] != 0


def assert_board_is_self_consistent(board):
    matrix = board._matrix
    tensor = board.tensor
    assert np.array_equal(matrix, tensor[0].cpu())



