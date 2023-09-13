import numpy as np

class GoBoard:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0=empty, 1=black, -1=white
        self.previous_board = None  # To check for Ko rule

    def possible_moves(self, color):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == 0:  # Empty intersection
                    if self.is_valid_move((x, y), color):  # Check if the move is valid
                        moves.append((x, y))
        moves.append('pass')  # You can always pass
        return moves

    def is_valid_move(self, position, color):
        x, y = position
        # Check for out-of-bounds
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False

        # Check if the intersection is empty
        if self.board[x, y] != 0:
            return False

        # Temporary place the stone to check for suicide rule
        self.board[x, y] = color
        if self.count_liberties(position) == 0:  # No suicide rule
            self.board[x, y] = 0  # Remove the temporarily placed stone
            return False

        # TODO: Check for Ko rule

        self.board[x, y] = 0  # Remove the temporarily placed stone
        return True

    def count_liberties(self, position):
        stack = [position]
        liberties = 0
        visited = set()

        color = self.board[position]

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size:
                    continue

                if self.board[nx, ny] == 0:
                    liberties += 1

                elif self.board[nx, ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

        return liberties
