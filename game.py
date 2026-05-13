import numpy as np
import os
import time
from PIL import Image
import matplotlib.colors as mcolors
from numba import njit
from generate_chessboards import generate_inverted_spiral, generate_raster, generate_snake, generate_spiral

def parse_color(c):
    """Parse color from string name, hex string, or (R,G,B[,A]) tuple (0-255)."""
    if isinstance(c, str):
        rgba = mcolors.to_rgba(c)
        return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
    elif isinstance(c, (tuple, list)):
        return (int(c[0]), int(c[1]), int(c[2]))
    return (255, 255, 255)

# ---------------------------------------------------------------------------
# Precompute leap offsets for all known leaper types.
# ---------------------------------------------------------------------------
def _leaper_offsets(dx, dy):
    offsets = set()
    for sx in (-1, 1):
        for sy in (-1, 1):
            offsets.add((dx * sx, dy * sy))
            offsets.add((dy * sx, dx * sy))
    return np.array(list(offsets), dtype=np.int32)

LEAPER_SPECS = {
    'wazir': (0, 1), 'ferz': (1, 1), 'dabbaba': (0, 2),
    'knight': (1, 2), 'alfil': (2, 2), 'camel': (1, 3),
    'zebra': (2, 3), 'tripper': (3, 3), 'fourleaper': (0, 4),
    'giraffe': (1, 4), 'stag': (2, 4), 'antelope': (3, 4),
    'threeleaper': (0, 3),
}

LEAPER_OFFSETS = {}
for name, spec in LEAPER_SPECS.items():
    LEAPER_OFFSETS[name] = _leaper_offsets(*spec)
# King = wazir + ferz
king_offs = set()
for sx in (-1, 1):
    for sy in (-1, 1):
        king_offs.add((0 * sx, 1 * sy))
        king_offs.add((1 * sx, 0 * sy))
        king_offs.add((1 * sx, 1 * sy))
LEAPER_OFFSETS['king'] = np.array(list(king_offs), dtype=np.int32)

# Compound pieces that also have knight leaps
COMPOUND_WITH_KNIGHT = {'dragon', 'amazon', 'empress', 'chancellor', 'princess', 'archbishop'}

# Sliding directions per piece type — encoded as bitmask for numba:
#   bit 0: ortho (-1,0)  bit 1: ortho (1,0)  bit 2: ortho (0,-1)  bit 3: ortho (0,1)
#   bit 4: diag (-1,-1)  bit 5: diag (-1,1)  bit 6: diag (1,-1)   bit 7: diag (1,1)
ORTHO_MASK = 0x0F   # bits 0-3
DIAG_MASK  = 0xF0   # bits 4-7
ALL8_MASK  = 0xFF   # bits 0-7

SLIDER_MASKS = {
    'rook': ORTHO_MASK, 'bishop': DIAG_MASK, 'queen': ALL8_MASK,
    'dragon': ALL8_MASK, 'amazon': ALL8_MASK,
    'empress': ORTHO_MASK, 'chancellor': ORTHO_MASK,
    'princess': DIAG_MASK, 'archbishop': DIAG_MASK,
}

# The 8 directions corresponding to bits 0-7
ALL8_DIRS = np.array([
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-1, -1], [-1, 1], [1, -1], [1, 1]
], dtype=np.int32)

def _piece_offsets(ptype):
    """Return leap offsets (excluding sliding rays) for a piece type."""
    ptype = ptype.lower()
    offsets_list = []
    if ptype in LEAPER_OFFSETS:
        offsets_list.append(LEAPER_OFFSETS[ptype])
    elif ptype.startswith('leaper_'):
        parts = ptype.split('_')
        if len(parts) == 3:
            a, b = int(parts[1]), int(parts[2])
            offsets_list.append(_leaper_offsets(a, b))
    if ptype in COMPOUND_WITH_KNIGHT:
        offsets_list.append(LEAPER_OFFSETS['knight'])
    if offsets_list:
        return np.vstack(offsets_list).astype(np.int32)
    return np.empty((0, 2), dtype=np.int32)

def _piece_slider_mask(ptype):
    """Return the slider bitmask for a piece type (0 if not a slider)."""
    return SLIDER_MASKS.get(ptype.lower(), 0)

# ---------------------------------------------------------------------------
# Numba-accelerated simulation kernel (with full slider support)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _simulate(N, num_players, search_rows, search_cols, total_cells,
              player_piece_counts,
              player_offsets_flat, player_offsets_starts, player_offsets_lens,
              player_slider_masks,
              all8_dirs, has_any_sliders):
    """
    Run the entire simulation in compiled code.
    Supports both leapers and sliding pieces with ray-blocking.

    Returns:
        occupied: int8[N, N] — player id per cell (-1 = empty)
        total_turns: int
    """
    occupied = np.full((N, N), -1, dtype=np.int8)
    control_count = np.zeros((num_players, N, N), dtype=np.int32)

    # For slider support: store the slider mask of the piece at each cell
    piece_slider_mask = np.zeros((N, N), dtype=np.int32)
    # Store which player owns each cell (redundant with occupied but avoids casts)
    piece_owner = np.full((N, N), -1, dtype=np.int32)

    # For slider games: store sequence number for each cell so we can track
    # the minimum possibly-freed square per player when rays are blocked
    seq_num = np.zeros((N, N), dtype=np.int64)
    for i in range(total_cells):
        seq_num[search_rows[i], search_cols[i]] = i

    search_indices = np.zeros(num_players, dtype=np.int64)
    active = np.ones(num_players, dtype=np.bool_)
    piece_indices = np.zeros(num_players, dtype=np.int64)

    total_turns = 0

    # History arrays (pre-allocate max possible size)
    hist_rows = np.empty(total_cells, dtype=np.int32)
    hist_cols = np.empty(total_cells, dtype=np.int32)
    hist_pids = np.empty(total_cells, dtype=np.int32)
    hist_pidxs = np.empty(total_cells, dtype=np.int64)

    while True:
        any_active = False
        for pid in range(num_players):
            if not active[pid]:
                continue
            any_active = True

            # Determine which piece in the cycle
            pidx = piece_indices[pid] % player_piece_counts[pid]

            # Get the offset array for this piece's leaps
            off_start = player_offsets_starts[pid, pidx]
            off_len = player_offsets_lens[pid, pidx]
            # Get the slider mask for this piece
            smask = player_slider_masks[pid, pidx]

            # Search for valid square
            idx = search_indices[pid]
            found = False

            while idx < total_cells:
                r = search_rows[idx]
                c = search_cols[idx]

                if occupied[r, c] == -1:
                    # Check enemy control
                    dominated = False
                    for p in range(num_players):
                        if p != pid and control_count[p, r, c] > 0:
                            dominated = True
                            break

                    if not dominated:
                        # --- Place piece ---
                        occupied[r, c] = pid
                        piece_owner[r, c] = pid
                        piece_slider_mask[r, c] = smask

                        # --- Handle ray blocking ---
                        # A newly placed piece can block rays from existing
                        # sliders.  For each of the 8 directions, walk backwards
                        # to find a slider whose ray passed through (r,c).
                        if has_any_sliders:
                            for d in range(8):
                                dr = all8_dirs[d, 0]
                                dc = all8_dirs[d, 1]
                                bit = 1 << d
                                # Walk backwards (opposite direction) to find blocker
                                nr = r - dr
                                nc = c - dc
                                while 0 <= nr < N and 0 <= nc < N:
                                    if occupied[nr, nc] != -1:
                                        owner = piece_owner[nr, nc]
                                        pmask = piece_slider_mask[nr, nc]
                                        if pmask & bit:
                                            # This piece slides in direction d.
                                            # Remove control beyond (r,c).
                                            br = r + dr
                                            bc = c + dc
                                            while 0 <= br < N and 0 <= bc < N:
                                                control_count[owner, br, bc] -= 1
                                                # If we just freed a square from
                                                # enemy control, a player's search
                                                # index might need to go back.
                                                freed_idx = seq_num[br, bc]
                                                for pp in range(num_players):
                                                    if pp != owner and search_indices[pp] > freed_idx:
                                                        search_indices[pp] = freed_idx
                                                if occupied[br, bc] != -1:
                                                    break
                                                br += dr
                                                bc += dc
                                        break  # stop at first piece in this direction
                                    nr -= dr
                                    nc -= dc

                        # --- Add leap control ---
                        for oi in range(off_len):
                            dr = player_offsets_flat[off_start + oi, 0]
                            dc = player_offsets_flat[off_start + oi, 1]
                            nr = r + dr
                            nc = c + dc
                            if 0 <= nr < N and 0 <= nc < N:
                                control_count[pid, nr, nc] += 1

                        # --- Add ray control ---
                        if smask != 0:
                            for d in range(8):
                                if smask & (1 << d):
                                    dr = all8_dirs[d, 0]
                                    dc = all8_dirs[d, 1]
                                    nr = r + dr
                                    nc = c + dc
                                    while 0 <= nr < N and 0 <= nc < N:
                                        control_count[pid, nr, nc] += 1
                                        if occupied[nr, nc] != -1:
                                            break
                                        nr += dr
                                        nc += dc

                        hist_rows[total_turns] = r
                        hist_cols[total_turns] = c
                        hist_pids[total_turns] = pid
                        hist_pidxs[total_turns] = pidx
                        total_turns += 1
                        piece_indices[pid] += 1
                        search_indices[pid] = idx + 1
                        found = True
                        break

                idx += 1

            if not found:
                search_indices[pid] = idx
                active[pid] = False

        if not any_active:
            break

    # Trim history arrays
    hist_r = hist_rows[:total_turns]
    hist_c = hist_cols[:total_turns]
    hist_p = hist_pids[:total_turns]
    hist_pi = hist_pidxs[:total_turns]
    return occupied, total_turns, hist_r, hist_c, hist_p, hist_pi

# ---------------------------------------------------------------------------
# High-level Game wrapper
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, N, board_sequence, players_config):
        self.N = N
        self.players = players_config
        self.num_players = len(players_config)

        # Precompute search order
        order = np.argsort(board_sequence, axis=None)
        self.search_rows = np.ascontiguousarray(order // N, dtype=np.int32)
        self.search_cols = np.ascontiguousarray(order % N, dtype=np.int32)

        # Build flattened offset arrays for numba
        max_pieces = max(len(p['pieces']) for p in players_config)
        player_piece_counts = np.array([len(p['pieces']) for p in players_config], dtype=np.int64)

        all_offsets = []
        starts = np.zeros((self.num_players, max_pieces), dtype=np.int64)
        lens = np.zeros((self.num_players, max_pieces), dtype=np.int64)
        slider_masks = np.zeros((self.num_players, max_pieces), dtype=np.int32)
        cursor = 0

        has_any_sliders = False
        for pid, p in enumerate(players_config):
            for pidx, pname in enumerate(p['pieces']):
                offs = _piece_offsets(pname)
                starts[pid, pidx] = cursor
                lens[pid, pidx] = len(offs)
                if len(offs) > 0:
                    all_offsets.append(offs)
                cursor += len(offs)

                sm = _piece_slider_mask(pname)
                slider_masks[pid, pidx] = sm
                if sm != 0:
                    has_any_sliders = True

        if all_offsets:
            flat = np.vstack(all_offsets).astype(np.int32)
        else:
            flat = np.empty((0, 2), dtype=np.int32)

        self.player_piece_counts = player_piece_counts
        self.player_offsets_flat = flat
        self.player_offsets_starts = starts
        self.player_offsets_lens = lens
        self.player_slider_masks = slider_masks
        self.has_any_sliders = has_any_sliders
        self.board_sequence = board_sequence
        self.occupied = None
        self.history = None  # (rows, cols, pids, pidxs)

    def run(self):
        """Run the full simulation using the numba-compiled kernel."""
        result = _simulate(
            self.N, self.num_players,
            self.search_rows, self.search_cols, self.N * self.N,
            self.player_piece_counts,
            self.player_offsets_flat, self.player_offsets_starts, self.player_offsets_lens,
            self.player_slider_masks,
            ALL8_DIRS, self.has_any_sliders
        )
        self.occupied = result[0]
        total_turns = result[1]
        self.history = (result[2], result[3], result[4], result[5])
        return total_turns

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def save_pixel_board(game, filename):
    """Render board as PNG, 1 pixel = 1 square. Fully vectorised."""
    N = game.N
    num_p = game.num_players
    lut = np.empty((num_p + 1, 3), dtype=np.uint8)
    lut[0] = [255, 255, 255]
    for i, p in enumerate(game.players):
        rgb = parse_color(p['color'])
        lut[i + 1] = rgb

    idx_map = game.occupied.astype(np.int16) + 1
    img_data = lut[idx_map.ravel()].reshape(N, N, 3)
    img = Image.fromarray(img_data, 'RGB')
    img.save(filename)
    print(f"Saved {filename}")


def plot_board(game, occupied, piece_names, title, filename,
               show_numbers=True, show_labels=True):
    """Render a board state using matplotlib with optional square numbers
    and piece labels.  Only practical for small boards (N <= ~30).

    Args:
        game: Game instance (used for N, board_sequence, players).
        occupied: int8[N, N] board state (-1 = empty).
        piece_names: object[N, N] array of piece name strings ('' = empty).
        title: plot title string.
        filename: output image path.
        show_numbers: if True, draw the sequence number on each square.
        show_labels: if True, draw the piece abbreviation on placed squares.
    """
    import matplotlib.pyplot as plt
    N = game.N
    fig, ax = plt.subplots(figsize=(max(6, N * 0.7), max(6, N * 0.7)))

    # Checkerboard
    board_img = np.zeros((N, N, 3), dtype=np.float32)
    for r in range(N):
        for c in range(N):
            pid = occupied[r, c]
            if pid != -1:
                rgb = parse_color(game.players[pid]['color'])
                board_img[r, c] = [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]
            elif (r + c) % 2 == 0:
                board_img[r, c] = [1.0, 1.0, 1.0]
            else:
                board_img[r, c] = [0.33, 0.33, 0.33]
    ax.imshow(board_img)

    # Text overlays
    fontsize = max(4, min(12, 80 // N))
    for r in range(N):
        for c in range(N):
            lines = []
            if show_numbers:
                lines.append(str(game.board_sequence[r, c]))
            if show_labels and piece_names[r, c] != '':
                lines.append(piece_names[r, c][:2].capitalize())
            if lines:
                pid = occupied[r, c]
                # Pick contrasting text color
                if pid != -1:
                    rgb = parse_color(game.players[pid]['color'])
                    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    tc = 'white' if lum < 128 else 'black'
                elif (r + c) % 2 == 0:
                    tc = 'black'
                else:
                    tc = 'white'
                ax.text(c, r, '\n'.join(lines), ha='center', va='center',
                        fontsize=fontsize, fontweight='bold', color=tc)

    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_animation(game, filename, show_numbers=True, show_labels=True,
                   duration=100):
    """Replay the game history into a GIF animation.
    Only practical for small boards (N <= ~30) and short games.

    Args:
        game: Game instance (must have been run with game.run() first).
        filename: output GIF path.
        show_numbers: draw sequence numbers on each square.
        show_labels: draw piece abbreviations on placed squares.
        duration: milliseconds per frame.
    """
    import matplotlib.pyplot as plt
    if game.history is None:
        raise RuntimeError("Call game.run() before save_animation()")

    hist_r, hist_c, hist_p, hist_pi = game.history
    N = game.N
    total_turns = len(hist_r)

    frames_dir = 'visualizations/frames'
    os.makedirs(frames_dir, exist_ok=True)

    occ = np.full((N, N), -1, dtype=np.int8)
    names = np.full((N, N), '', dtype=object)
    frame_files = []

    # Frame 0: empty board
    fn = f'{frames_dir}/frame_000.png'
    plot_board(game, occ, names, 'Start', fn, show_numbers, show_labels)
    frame_files.append(fn)

    for t in range(total_turns):
        r, c = int(hist_r[t]), int(hist_c[t])
        pid = int(hist_p[t])
        pidx = int(hist_pi[t])
        piece_name = game.players[pid]['pieces'][pidx]
        occ[r, c] = pid
        names[r, c] = piece_name

        fn = f'{frames_dir}/frame_{t+1:03d}.png'
        title = f'Turn {t+1}: P{pid+1} placed {piece_name} at ({r},{c})'
        plot_board(game, occ, names, title, fn, show_numbers, show_labels)
        frame_files.append(fn)

    # Hold last frame longer
    for _ in range(5):
        frame_files.append(frame_files[-1])

    images = [Image.open(f) for f in frame_files]
    images[0].save(filename, save_all=True, append_images=images[1:],
                   duration=duration, loop=0)
    print(f"Saved animation: {filename} ({len(frame_files)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    N = 1000

    # --- Options ---
    ANIMATE = False          # Generate step-by-step GIF (small boards only)
    SHOW_NUMBERS = False     # Draw sequence numbers on squares
    SHOW_LABELS = False      # Draw piece abbreviations on placed squares
    PIXEL_MODE = True       # Also save high-res 1px-per-square image

    t0 = time.time()
    print(f"Generating {N}x{N} board...")
    sequence = generate_inverted_spiral(N)
    print(f"  Spiral: {time.time()-t0:.2f}s")

    players = [
        {'color': '#E2725B', 'pieces': ['Knight', ]},
        {'color': '#8A9A5B', 'pieces': [ 'stag', 'king' ]},
        {'color': '#DCAE96', 'pieces': [ 'ferz', 'wazir' ]},
    ]

    t0 = time.time()
    print("Initializing Game...")
    game = Game(N, sequence, players)
    print(f"  Init: {time.time()-t0:.2f}s")

    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

    # Warm up numba JIT
    print("JIT compiling simulation kernel (first run only)...")
    t0 = time.time()
    _w = Game(8, generate_inverted_spiral(8), players)
    _w.run()
    print(f"  JIT compile: {time.time()-t0:.2f}s")

    t0 = time.time()
    print(f"Running {N}x{N} simulation...")
    total_turns = game.run()
    elapsed = time.time() - t0
    print(f"  Simulation complete: {total_turns:,} turns in {elapsed:.1f}s")

    if PIXEL_MODE:
        save_pixel_board(game, f"{out_dir}/game_final_board_pixel.png")

    if ANIMATE:
        save_animation(game, f"{out_dir}/game_animation.gif",
                       show_numbers=SHOW_NUMBERS, show_labels=SHOW_LABELS)

    # Also save final board as annotated matplotlib image
    if SHOW_NUMBERS or SHOW_LABELS:
        hist_r, hist_c, hist_p, hist_pi = game.history
        names = np.full((N, N), '', dtype=object)
        for t in range(total_turns):
            r, c = int(hist_r[t]), int(hist_c[t])
            pid = int(hist_p[t])
            pidx = int(hist_pi[t])
            names[r, c] = game.players[pid]['pieces'][pidx]
        plot_board(game, game.occupied, names, 'Final Board',
                   f'{out_dir}/game_final_board.png',
                   SHOW_NUMBERS, SHOW_LABELS)

if __name__ == "__main__":
    main()
