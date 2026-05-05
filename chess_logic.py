import copy

# Piece types
PAWN = 'pawn'
ROOK = 'rook'
KNIGHT = 'knight'
BISHOP = 'bishop'
QUEEN = 'queen'
KING = 'king'

# Colors
WHITE = 'white'
BLACK = 'black'

class Piece:
    def __init__(self, type, color):
        self.type = type
        self.color = color

    def __str__(self):
        symbol = {'pawn': 'P', 'rook': 'R', 'knight': 'N', 'bishop': 'B', 'queen': 'Q', 'king': 'K'}[self.type]
        return symbol if self.color == WHITE else symbol.lower()

class Board:
    def __init__(self):
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        self.setup_pieces()

    def setup_pieces(self):
        # White pieces
        self.squares[0] = [Piece(ROOK, WHITE), Piece(KNIGHT, WHITE), Piece(BISHOP, WHITE), Piece(QUEEN, WHITE),
                           Piece(KING, WHITE), Piece(BISHOP, WHITE), Piece(KNIGHT, WHITE), Piece(ROOK, WHITE)]
        self.squares[1] = [Piece(PAWN, WHITE) for _ in range(8)]
        # Black pieces
        self.squares[6] = [Piece(PAWN, BLACK) for _ in range(8)]
        self.squares[7] = [Piece(ROOK, BLACK), Piece(KNIGHT, BLACK), Piece(BISHOP, BLACK), Piece(QUEEN, BLACK),
                           Piece(KING, BLACK), Piece(BISHOP, BLACK), Piece(KNIGHT, BLACK), Piece(ROOK, BLACK)]

    def get_piece(self, row, col):
        if 0 <= row < 8 and 0 <= col < 8:
            return self.squares[row][col]
        return None

    def set_piece(self, row, col, piece):
        if 0 <= row < 8 and 0 <= col < 8:
            self.squares[row][col] = piece

    def move_piece(self, from_row, from_col, to_row, to_col):
        piece = self.get_piece(from_row, from_col)
        if piece:
            self.set_piece(from_row, from_col, None)
            self.set_piece(to_row, to_col, piece)
            return True
        return False

    def is_empty(self, row, col):
        return self.get_piece(row, col) is None

    def is_enemy(self, row, col, color):
        piece = self.get_piece(row, col)
        return piece and piece.color != color

    def __str__(self):
        board_str = ""
        for row in range(7, -1, -1):  # Print from rank 8 to 1
            board_str += str(row + 1) + " "
            for col in range(8):
                piece = self.squares[row][col]
                board_str += (str(piece) if piece else '.') + " "
            board_str += "\n"
        board_str += "  a b c d e f g h\n"
        return board_str

class ChessGame:
    def __init__(self):
        self.boards = [Board() for _ in range(4)]  # 0: right, 1: left, 2: top, 3: bottom
        self.current_turn = WHITE
        self.selected = None  # (board_idx, row, col)
        self.wall_names = ["Right", "Left", "Top", "Bottom"]

    def select_square(self, board_idx, row, col):
        if self.selected is None:
            piece = self.boards[board_idx].get_piece(row, col)
            if piece and piece.color == self.current_turn:
                self.selected = (board_idx, row, col)
                print(f"Player 1 selected: {self.wall_names[board_idx]} {chr(col+97)}{row+1}")
                return True
            else:
                print("No piece or not your turn")
                return False
        else:
            # Attempt move
            from_board, from_row, from_col = self.selected
            to_board, to_row, to_col = board_idx, row, col
            if self.is_valid_move(from_board, from_row, from_col, to_board, to_row, to_col):
                self.move_piece(from_board, from_row, from_col, to_board, to_row, to_col)
                print(f"Player 1: {self.wall_names[from_board]} {chr(from_col+97)}{from_row+1} to {self.wall_names[to_board]} {chr(to_col+97)}{to_row+1}")
                self.selected = None
                self.switch_turn()
                return True
            else:
                print("Invalid move")
                self.selected = None
                return False

    def is_valid_move(self, from_board, from_row, from_col, to_board, to_row, to_col):
        if from_board == to_board and from_row == to_row and from_col == to_col:
            return False  # Can't move to the same square
        board = self.boards[from_board]
        piece = board.get_piece(from_row, from_col)
        if not piece:
            return False
        if from_board == to_board:
            # Same board, check standard chess rules (simplified)
            return self.is_valid_same_board(piece, from_row, from_col, to_row, to_col, board)
        else:
            # Inter-board move, allow if legal on source board but destination on another
            # For wildcard, allow any move to any board if the piece can reach the edge in that direction
            # Simplified: allow if the move is sliding to the edge
            if piece.type in [ROOK, BISHOP, QUEEN, KNIGHT]:
                return self.can_slide_to_edge(piece, from_row, from_col, to_board, to_row, to_col, board)
            else:
                return False  # Only sliding pieces for inter-board

    def is_valid_same_board(self, piece, from_row, from_col, to_row, to_col, board):
        # Simplified validation, only check if destination is empty or enemy
        if board.is_empty(to_row, to_col) or board.is_enemy(to_row, to_col, piece.color):
            # For now, no path checking, just allow
            return True
        return False

    def can_slide_to_edge(self, piece, from_row, from_col, to_board, to_row, to_col, board):
        if piece.type == 'knight':
            return to_board != from_board  # Knights can "jump" to any other board
        # Check if the move is towards the edge and path is clear
        dr = to_row - from_row
        dc = to_col - from_col
        if dr == 0:  # horizontal
            if dc > 0 and from_col < 7:  # right
                for c in range(from_col + 1, 8):
                    if not board.is_empty(from_row, c):
                        return False
                return to_board == 1 and to_col == 0  # to left board, col 0
            elif dc < 0 and from_col > 0:  # left
                for c in range(from_col - 1, -1, -1):
                    if not board.is_empty(from_row, c):
                        return False
                return to_board == 0 and to_col == 7  # to right board, col 7
        elif dc == 0:  # vertical
            if dr > 0 and from_row < 7:  # up
                for r in range(from_row + 1, 8):
                    if not board.is_empty(r, from_col):
                        return False
                return to_board == 3 and to_row == 0  # to bottom board, row 0
            elif dr < 0 and from_row > 0:  # down
                for r in range(from_row - 1, -1, -1):
                    if not board.is_empty(r, from_col):
                        return False
                return to_board == 2 and to_row == 7  # to top board, row 7
        # For bishop, diagonal to edge
        if abs(dr) == abs(dc):
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1
            r, c = from_row + step_r, from_col + step_c
            while r != to_row and c != to_col:
                if not board.is_empty(r, c):
                    return False
                r += step_r
                c += step_c
            # Check if reaching edge
            if (dr > 0 and from_row < 7 and dc > 0 and from_col < 7 and to_board == 3 and to_row == 0 and to_col == 0) or \
               (dr > 0 and from_row < 7 and dc < 0 and from_col > 0 and to_board == 3 and to_row == 0 and to_col == 7) or \
               (dr < 0 and from_row > 0 and dc > 0 and from_col < 7 and to_board == 2 and to_row == 7 and to_col == 0) or \
               (dr < 0 and from_row > 0 and dc < 0 and from_col > 0 and to_board == 2 and to_row == 7 and to_col == 7):
                return True
        return False

    def move_piece(self, from_board, from_row, from_col, to_board, to_row, to_col):
        piece = self.boards[from_board].get_piece(from_row, from_col)
        if piece:
            self.boards[from_board].set_piece(from_row, from_col, None)
            self.boards[to_board].set_piece(to_row, to_col, piece)

    def switch_turn(self):
        self.current_turn = BLACK if self.current_turn == WHITE else WHITE
        print(f"Turn: {self.current_turn}")

    def print_boards(self):
        board_names = ['Right', 'Left', 'Top', 'Bottom']
        for i, board in enumerate(self.boards):
            print(f"\n{board_names[i]} Board:")
            print(board)

# Example usage
if __name__ == "__main__":
    game = ChessGame()
    game.print_boards()