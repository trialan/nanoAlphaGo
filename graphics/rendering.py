import arcade
import numpy as np

class GoBoardWindow(arcade.Window):
    def __init__(self, board_array, window_size=500):
        super().__init__(window_size, window_size, "Go Board")
        self.board_array = np.transpose(board_array[::-1])  # Transpose the array
        self.window_size = window_size
        self.grid_size = self.board_array.shape[0]
        self.cell_size = self.window_size // (self.grid_size + 1)

    def on_draw(self):
        arcade.start_render()

        # Set up colors and sizes
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BOARD_COLOR = (220, 179, 92)

        # Draw board background
        arcade.draw_lrtb_rectangle_filled(0, self.window_size, self.window_size, 0, BOARD_COLOR)

        # Draw grid lines
        for i in range(1, self.grid_size + 1):
            arcade.draw_line(i * self.cell_size, self.cell_size, i * self.cell_size, self.window_size - self.cell_size, BLACK, 1)
            arcade.draw_line(self.cell_size, i * self.cell_size, self.window_size - self.cell_size, i * self.cell_size, BLACK, 1)

        # Draw stones
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = (i + 1) * self.cell_size, (j + 1) * self.cell_size  # Adjusted for top-left origin
                if self.board_array[i, j] == 1:
                    arcade.draw_circle_filled(x, y, self.cell_size // 2 - 2, BLACK)
                elif self.board_array[i, j] == -1:
                    arcade.draw_circle_filled(x, y, self.cell_size // 2 - 2, WHITE)

def display_board(board):
    board_array = board.board
    window = GoBoardWindow(board_array)
    arcade.run()

if __name__ == "__main__":
    # Create a 9x9 board
    board = np.zeros((9, 9), dtype=int)
    board[0, 0] = 1  # Black stone at (0, 0)
    board[1, 1] = -1  # White stone at (1, 1)
    display_board(board)
