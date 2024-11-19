import pygame

class TicTacToeDisplay:
    def __init__(self, rows, cols, cell_size=100):
        """
        Handles rendering of the Tic Tac Toe game board.

        :param rows: Number of rows on the board.
        :param cols: Number of columns on the board.
        :param cell_size: Size of each cell in pixels.
        """
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.width = cols * cell_size
        self.height = rows * cell_size

        # Colors
        self.bg_color = (28, 170, 156)
        self.line_color = (23, 145, 135)
        self.circle_color = (239, 231, 200)
        self.cross_color = (84, 84, 84)
        self.line_width = 5
        self.circle_width = 8
        self.cross_width = 10
        self.space = 20

        # Initialize Pygame screen
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")

    def render(self, board_state):
        """
        Renders the game board with the given board state.

        :param board_state: A 2D list representing the current game state.
        """
        self.screen.fill(self.bg_color)

        # Draw grid lines
        for row in range(1, self.rows):
            pygame.draw.line(self.screen, self.line_color,
                             (0, row * self.cell_size), (self.width, row * self.cell_size), self.line_width)
        for col in range(1, self.cols):
            pygame.draw.line(self.screen, self.line_color,
                             (col * self.cell_size, 0), (col * self.cell_size, self.height), self.line_width)

        # Draw X's and O's
        for row in range(self.rows):
            for col in range(self.cols):
                if board_state[row][col] == 'X':
                    pygame.draw.line(self.screen, self.cross_color,
                                     (col * self.cell_size + self.space, row * self.cell_size + self.space),
                                     ((col + 1) * self.cell_size - self.space,
                                      (row + 1) * self.cell_size - self.space), self.cross_width)
                    pygame.draw.line(self.screen, self.cross_color,
                                     (col * self.cell_size + self.space,
                                      (row + 1) * self.cell_size - self.space),
                                     ((col + 1) * self.cell_size - self.space, row * self.cell_size + self.space),
                                     self.cross_width)
                elif board_state[row][col] == 'O':
                    pygame.draw.circle(self.screen, self.circle_color,
                                       (col * self.cell_size + self.cell_size // 2,
                                        row * self.cell_size + self.cell_size // 2),
                                       self.cell_size // 3, self.circle_width)

        pygame.display.update()
