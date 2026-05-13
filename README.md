# Chessboard Sequences Simulation

A high-performance Python engine for simulating piece placement games on arbitrary chessboard layouts (up to 10,000 x 10,000 and beyond).

The game engine simulates a game where players take turns placing pieces on a board. Pieces are placed on the lowest available numbered square that is not attacked by the opponent. The game supports complex piece movement rules (including sliding pieces and leapers) and can handle multiple players.

## Features

- **Extreme Scale**: Powered by a heavily optimized Numba JIT-compiled engine capable of simulating 10,000x10,000 boards (100 million squares) in seconds.
- **$O(1)$ Amortized Search**: Replaces standard board scans with an amortized search index for instant piece placement.
- **Complex Pieces**: Supports both standard chess pieces and a vast array of Fairy Chess pieces:
    - **Sliders**: Rook, Bishop, Queen, Dragon, Amazon, Empress, Chancellor, Princess, Archbishop.
    - **Leapers**: Wazir, Ferz, Dabbaba, Knight, Alfil, Camel, Zebra, Tripper, Fourleaper, Giraffe, Stag, Antelope, Threeleaper.
    - **Custom Leapers**: Specify any piece dynamically using `'leaper_A_B'` format.
- **Multi-Player Support**: Define any number of players, each with their own color and piece sequences.
- **Ray-Blocking Engine**: Automatically handles the truncation of sliding piece attacks when a new piece is placed.
- **Multiple Sequence Generators**: Raster, Snake, Archimedean Spiral, and Inverted Spiral.
- **Visualizations**: 
    - Standard `matplotlib` annotated boards for small scales.
    - Step-by-step GIF animations for studying patterns.
    - Direct-to-image High-Resolution Pixel mode (1 pixel = 1 square) for massive boards, completely bypassing memory limitations.

## Generating Piece Move Diagrams
To understand how different pieces control the board, run the piece diagram generator:
```bash
python generate_piece_moves.py
```
This will output visualizations for all available pieces into the `piece_moves/` directory.

## Testing & Verification
The repository includes a pure Python reference implementation used to rigorously verify the optimized Numba kernel across both leaper and slider scenarios:
```bash
python verify.py
```

## Running the Engine
Simply define your players and board size in `game.py` and run:
```bash
python game.py
```
Outputs are automatically saved to the `visualizations/` directory.
