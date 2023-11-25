import numpy as np

from nanoAlphaGo.config import BOARD_SIZE, PASS
from nanoAlphaGo.rl.utils import _index_to_move, _nn_tensor_from_matrix

"""
Bad things about this code:
    - It's confusing to have both self._matrix and self.tensor,
      ideally we'd only have self.tensor.
"""

class GoBoard:
    def __init__(self, size=BOARD_SIZE, initial_state_matrix=None):
        self.size = size
        if initial_state_matrix is None:
            self._matrix = np.zeros((size, size), dtype=int)
        else:
            self._matrix = initial_state_matrix
        self.tensor = _nn_tensor_from_matrix(self._matrix)
        self.previous_board = None  # To check for Ko rule

    def legal_moves(self, color):
        empty_positions = np.column_stack(np.where(self._matrix == 0))
        moves = [(x, y) for x, y in empty_positions if self.is_valid_move((x, y), color)]
        moves.append(PASS)
        return moves

    def is_valid_move(self, position, color):
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
            if self.count_liberties(position) == 0:
                valid = False
            self._matrix[x, y] = 0  # Reset the position to its original state

        # TODO: Check for Ko rule
        return valid

    def count_liberties(self, position):
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
        return n_unique_liberties

    def apply_move(self, move, color):
        move = _index_to_move(move)
        assert self.is_valid_move(move, color)

        if move == PASS:
            return

        x, y = move
        self._matrix[x, y] = color

        opponent = -color
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                continue
            if self._matrix[nx, ny] == opponent:
                if self.count_liberties((nx, ny)) == 0:
                    captured_stones = self._remove_group((nx, ny))
        self.tensor = _nn_tensor_from_matrix(self._matrix)

    def _remove_group(self, position):
        stack = [position]
        color = self._matrix[position]
        captured_count = 0

        while stack:
            x, y = stack.pop()

            if self._matrix[x, y] == color:
                self._matrix[x, y] = 0  # Remove the stone
                captured_count += 1  # Increment the number of captured stones

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy

                    if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                        continue

                    stack.append((nx, ny))
        return captured_count
