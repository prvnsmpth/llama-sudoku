import random
import copy
import sys
import io
import json

random.seed(1337)

# Constants for a 4x4 sudoku
N = 4
BLOCK_SIZE = 2

def print_grid(grid, out=sys.stdout):
    # Print the top border
    print("+" + "---+" * N, file=out)
    
    # Print each row with vertical borders
    for row in grid:
        print("|", end="", file=out)
        for val in row:
            cell_value = str(val) if val != 0 else '.'
            print(f" {cell_value} |", end="", file=out)
        print(file=out)  # New line after each row
        print("+" + "---+" * N, file=out)  # Horizontal border between rows

def find_empty(grid):
    """Find the first empty cell (denoted by 0) in the grid."""
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0:
                return (i, j)
    return None

def is_valid(grid, row, col, num):
    """Check if num can be placed at grid[row][col] without violating sudoku rules."""
    # Check row and column
    for i in range(N):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    # Check 2x2 block
    start_row = (row // BLOCK_SIZE) * BLOCK_SIZE
    start_col = (col // BLOCK_SIZE) * BLOCK_SIZE
    for i in range(start_row, start_row + BLOCK_SIZE):
        for j in range(start_col, start_col + BLOCK_SIZE):
            if grid[i][j] == num:
                return False
    return True

def solve_sudoku(grid, solutions, limit=2):
    """
    Solve the sudoku using backtracking.
    Appends found solutions to the 'solutions' list.
    Stops early if the number of solutions reaches the given limit.
    """
    if len(solutions) >= limit:
        return

    empty = find_empty(grid)
    if not empty:
        # No empty cells: a complete solution is found.
        solutions.append(copy.deepcopy(grid))
        return

    row, col = empty
    for num in range(1, N + 1):
        if is_valid(grid, row, col, num):
            grid[row][col] = num
            solve_sudoku(grid, solutions, limit)
            grid[row][col] = 0

def count_solutions(grid, limit=2):
    """Return the number of solutions found for the given grid (up to the limit)."""
    solutions = []
    solve_sudoku(grid, solutions, limit)
    return len(solutions)

def generate_full_grid():
    """Generate a fully solved 4x4 sudoku grid using backtracking."""
    grid = [[0 for _ in range(N)] for _ in range(N)]
    
    def backtrack():
        empty = find_empty(grid)
        if not empty:
            return True  # A full solution has been constructed.
        row, col = empty
        numbers = list(range(1, N + 1))
        random.shuffle(numbers)
        for num in numbers:
            if is_valid(grid, row, col, num):
                grid[row][col] = num
                if backtrack():
                    return True
                grid[row][col] = 0
        return False

    backtrack()
    return grid

def generate_candidate_puzzle(full_grid, clue_count=6):
    """
    Given a full solution grid, randomly select 'clue_count' cells as clues.
    Returns a puzzle with the remaining cells set to 0.
    """
    indices = [(i, j) for i in range(N) for j in range(N)]
    clues = random.sample(indices, clue_count)
    puzzle = [[0 for _ in range(N)] for _ in range(N)]
    for (i, j) in clues:
        puzzle[i][j] = full_grid[i][j]
    return puzzle

def generate():
    # Generate a complete solution first.
    full_grid = generate_full_grid()
    print("Full Solution:")

    solution = io.StringIO()
    print_grid(full_grid, out=solution)

    attempts = 0
    # Try generating candidate puzzles until one has a unique solution.
    while True:
        candidate = generate_candidate_puzzle(full_grid, clue_count=8)
        # Use a deep copy to avoid accidental modifications during solving.
        sol_count = count_solutions(copy.deepcopy(candidate), limit=2)
        attempts += 1
        if sol_count == 1:
            print(f"Found a unique 8-clue puzzle after {attempts} attempts:")
            puzzle = io.StringIO()
            print_grid(candidate, out=puzzle)
            return [solution.getvalue(), puzzle.getvalue()]
        if attempts % 1000 == 0:
            print(f"Attempts so far: {attempts}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python generate.py <number of puzzles> <output file>")
        sys.exit(1)
    num_puzzles = int(sys.argv[1])
    output_file = sys.argv[2]
    with open(output_file, 'w') as f:
        for i in range(num_puzzles):
            solution, puzzle = generate()
            example = {
                'puzzle': puzzle,
                'solution': solution,
            }
            f.write(json.dumps(example) + '\n')
