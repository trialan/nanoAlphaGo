import numpy as np

from nanoAlphaGo.config import BOARD_SIZE, PASS
from nanoAlphaGo.rl.utils import _index_to_move

class GoBoard:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.matrix = np.zeros((size, size), dtype=int)
        self.previous_board = None  # To check for Ko rule

    def legal_moves(self, color):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.matrix[x, y] == 0:
                    if self.is_valid_move((x, y), color):
                        moves.append((x, y))
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
        elif self.matrix[x, y] != 0:
            valid = False

        else:
            # Temporary place the stone to check for suicide rule
            self.matrix[x, y] = color
            if self.count_liberties(position) == 0:
                valid = False
            self.matrix[x, y] = 0  # Reset the position to its original state

        # TODO: Check for Ko rule
        return valid

    def count_liberties(self, position):
        stack = [position]
        liberties_set = set()
        visited = set()

        color = self.matrix[position]

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                    continue

                if self.matrix[nx, ny] == 0:
                    liberties_set.add((nx, ny))

                elif self.matrix[nx, ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

        n_unique_liberties = len(liberties_set)
        return n_unique_liberties

    def apply_move(self, move, color):
        move = _index_to_move(move)
        assert self.is_valid_move(move, color)

        if move == PASS:
            return

        x, y = move
        self.matrix[x, y] = color

        opponent = -color
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                continue
            if self.matrix[nx, ny] == opponent:
                if self.count_liberties((nx, ny)) == 0:
                    captured_stones = self._remove_group((nx, ny))

    def _remove_group(self, position):
        stack = [position]
        color = self.matrix[position]
        captured_count = 0

        while stack:
            x, y = stack.pop()

            if self.matrix[x, y] == color:
                self.matrix[x, y] = 0  # Remove the stone
                captured_count += 1  # Increment the number of captured stones

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy

                    if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                        continue

                    stack.append((nx, ny))
        return captured_count
