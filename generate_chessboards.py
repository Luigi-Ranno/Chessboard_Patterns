import numpy as np
import matplotlib.pyplot as plt
import os

def generate_raster(N):
    """Row by row (top left one, increasing to right)"""
    return np.arange(1, N*N + 1).reshape((N, N))

def generate_snake(N):
    """Row by row, but continuously (left to right, then right to left)"""
    matrix = np.arange(1, N*N + 1).reshape((N, N))
    matrix[1::2, :] = matrix[1::2, ::-1]
    return matrix

def generate_spiral(N):
    """Archimedean spiral from top-left, inwards.
    Vectorized: builds the spiral by assigning contiguous slices per layer."""
    matrix = np.empty((N, N), dtype=np.int32)
    top, bottom, left, right = 0, N - 1, 0, N - 1
    num = 1
    while top <= bottom and left <= right:
        # Top row
        length = right - left + 1
        matrix[top, left:right+1] = np.arange(num, num + length)
        num += length
        top += 1

        # Right column
        length = bottom - top + 1
        if length > 0:
            matrix[top:bottom+1, right] = np.arange(num, num + length)
            num += length
        right -= 1

        if top <= bottom:
            # Bottom row (reversed)
            length = right - left + 1
            matrix[bottom, left:right+1] = np.arange(num + length - 1, num - 1, -1)
            num += length
            bottom -= 1

        if left <= right:
            # Left column (reversed)
            length = bottom - top + 1
            if length > 0:
                matrix[top:bottom+1, left] = np.arange(num + length - 1, num - 1, -1)
                num += length
            left += 1

    return matrix

def generate_inverted_spiral(N):
    """Archimedean spiral from center outwards. 1 is at the center.
    Uses the fast spiral and inverts the numbering."""
    spiral = generate_spiral(N)
    total = N * N
    return total + 1 - spiral

def plot_chessboard(matrix, title, filename):
    N = matrix.shape[0]
    
    # Create checkerboard pattern
    # 0 for white, 1 for black
    checkerboard = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i + j) % 2 == 1:
                checkerboard[i, j] = 1 

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot checkerboard using gray_r (0 is white, 1 is black)
    # Give it a slightly lighter black to keep text readable
    cmap = plt.cm.colors.ListedColormap(['#FFFFFF', '#555555'])
    ax.imshow(checkerboard, cmap=cmap) 
    
    # Add numbers
    for i in range(N):
        for j in range(N):
            # Text color depends on square color
            text_color = 'black' if checkerboard[i, j] == 0 else 'white'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', 
                    fontsize=12, fontweight='bold', color=text_color)
                    
    # Formatting
    ax.set_xticks(np.arange(-.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == '__main__':
    N = 8 # Standard chessboard size
    
    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Raster Sequence
    raster_mat = generate_raster(N)
    plot_chessboard(raster_mat, "Raster Sequence", os.path.join(out_dir, "raster.png"))
    
    # 2. Continuous Snake Sequence
    snake_mat = generate_snake(N)
    plot_chessboard(snake_mat, "Continuous Snake Sequence", os.path.join(out_dir, "snake.png"))
    
    # 3. Spiral Sequence
    spiral_mat = generate_spiral(N)
    plot_chessboard(spiral_mat, "Spiral Sequence", os.path.join(out_dir, "spiral.png"))
    
    # 4. Inverted Spiral Sequence
    inverted_spiral_mat = generate_inverted_spiral(N)
    plot_chessboard(inverted_spiral_mat, "Inverted Spiral Sequence", os.path.join(out_dir, "inverted_spiral.png"))
