import os
import numpy as np
import matplotlib.pyplot as plt
from game import _piece_offsets, _piece_slider_mask, LEAPER_SPECS, SLIDER_MASKS, ALL8_DIRS

def plot_piece_moves(piece_name, board_size=15):
    """
    Generate an image showing how a piece moves.
    Places the piece in the center of the board and highlights controlled squares.
    """
    piece_name_lower = piece_name.lower()
    
    # Create empty board
    N = board_size
    center_r, center_c = N // 2, N // 2
    
    # Checkerboard
    board_img = np.zeros((N, N, 3), dtype=np.float32)
    for r in range(N):
        for c in range(N):
            if (r + c) % 2 == 0:
                board_img[r, c] = [1.0, 1.0, 1.0]
            else:
                board_img[r, c] = [0.6, 0.6, 0.6]
                
    # Get piece moves
    offs = _piece_offsets(piece_name_lower)
    smask = _piece_slider_mask(piece_name_lower)
    
    controlled = set()
    
    # Leaps
    if len(offs) > 0:
        for i in range(len(offs)):
            dr, dc = int(offs[i, 0]), int(offs[i, 1])
            nr, nc = center_r + dr, center_c + dc
            if 0 <= nr < N and 0 <= nc < N:
                controlled.add((nr, nc))
                
    # Rays
    if smask != 0:
        for d in range(8):
            if smask & (1 << d):
                dr, dc = int(ALL8_DIRS[d, 0]), int(ALL8_DIRS[d, 1])
                nr, nc = center_r + dr, center_c + dc
                while 0 <= nr < N and 0 <= nc < N:
                    controlled.add((nr, nc))
                    nr += dr
                    nc += dc
                    
    # Highlight controlled squares
    highlight_color = [0.8, 0.2, 0.2]  # Red
    for (r, c) in controlled:
        # Blend highlight with board
        board_img[r, c] = 0.5 * board_img[r, c] + 0.5 * np.array(highlight_color)
        
    # Place piece in center
    piece_color = [0.2, 0.2, 0.8] # Blue
    board_img[center_r, center_c] = piece_color
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(board_img)
    
    # Add text
    ax.text(center_c, center_r, piece_name[:2].capitalize(), ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
            
    # Highlight points text 'X'
    for (r, c) in controlled:
        ax.text(c, r, 'X', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(f"Moves of {piece_name.capitalize()}", fontsize=16)
    
    out_dir = "piece_moves"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{piece_name_lower}.png")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Generated {filename}")

if __name__ == "__main__":
    # Get all pieces
    pieces = list(LEAPER_SPECS.keys()) + list(SLIDER_MASKS.keys()) + ['king']
    # remove duplicates
    pieces = list(set(pieces))
    
    for piece in pieces:
        plot_piece_moves(piece)
