"""Test sliding pieces against brute-force reference."""
import numpy as np
from generate_chessboards import generate_inverted_spiral
from game import Game, _piece_offsets, LEAPER_SPECS

# First, check what offsets sliding pieces get
print("=== Offset check ===")
for p in ['rook', 'bishop', 'queen', 'dragon']:
    offs = _piece_offsets(p)
    print(f"  {p}: {len(offs)} leap offsets")

print()

# Brute-force with full slider support
def brute_force_simulate(N, sequence, players_config):
    """Reference implementation with full sliding piece support."""
    num_players = len(players_config)
    occupied = {}
    piece_at = {}
    
    cells = []
    for r in range(N):
        for c in range(N):
            cells.append((sequence[r, c], r, c))
    cells.sort()
    search_order = [(r, c) for _, r, c in cells]
    
    SLIDER_DIRS = {
        'rook': [(-1,0),(1,0),(0,-1),(0,1)],
        'bishop': [(-1,-1),(-1,1),(1,-1),(1,1)],
        'queen': [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)],
        'dragon': [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)],
        'amazon': [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)],
        'empress': [(-1,0),(1,0),(0,-1),(0,1)],
        'chancellor': [(-1,0),(1,0),(0,-1),(0,1)],
        'princess': [(-1,-1),(-1,1),(1,-1),(1,1)],
        'archbishop': [(-1,-1),(-1,1),(1,-1),(1,1)],
    }
    
    LEAPER_OFFSETS_MAP = {}
    for name, spec in LEAPER_SPECS.items():
        offs = set()
        dx, dy = spec
        for sx in (-1, 1):
            for sy in (-1, 1):
                offs.add((dx*sx, dy*sy))
                offs.add((dy*sx, dx*sy))
        LEAPER_OFFSETS_MAP[name] = list(offs)
    # King
    king_offs = set()
    for sx in (-1,1):
        for sy in (-1,1):
            king_offs.add((0*sx, 1*sy)); king_offs.add((1*sx, 0*sy)); king_offs.add((1*sx, 1*sy))
    LEAPER_OFFSETS_MAP['king'] = list(king_offs)
    
    COMPOUND_KNIGHT = {'dragon','amazon','empress','chancellor','princess','archbishop'}
    
    def compute_controlled(player_id):
        controlled = set()
        for (r, c), pid in occupied.items():
            if pid != player_id:
                continue
            ptype = piece_at[(r, c)]
            
            # Leap offsets
            if ptype in LEAPER_OFFSETS_MAP:
                for dr, dc in LEAPER_OFFSETS_MAP[ptype]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        controlled.add((nr, nc))
            
            # Compound knight
            if ptype in COMPOUND_KNIGHT:
                for dr, dc in LEAPER_OFFSETS_MAP['knight']:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        controlled.add((nr, nc))
            
            # Sliding rays
            if ptype in SLIDER_DIRS:
                for dr, dc in SLIDER_DIRS[ptype]:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < N and 0 <= nc < N:
                        controlled.add((nr, nc))
                        if (nr, nc) in occupied:
                            break  # blocked
                        nr += dr
                        nc += dc
        
        return controlled
    
    active = [True] * num_players
    piece_indices = [0] * num_players
    total_turns = 0
    
    while any(active):
        for pid in range(num_players):
            if not active[pid]:
                continue
            plist = players_config[pid]['pieces']
            pidx = piece_indices[pid] % len(plist)
            piece = plist[pidx].lower()
            
            enemy_controlled = set()
            for p in range(num_players):
                if p != pid:
                    enemy_controlled |= compute_controlled(p)
            
            placed = False
            for r, c in search_order:
                if (r, c) not in occupied and (r, c) not in enemy_controlled:
                    occupied[(r, c)] = pid
                    piece_at[(r, c)] = piece
                    total_turns += 1
                    piece_indices[pid] += 1
                    placed = True
                    break
            if not placed:
                active[pid] = False
    
    result = np.full((N, N), -1, dtype=np.int8)
    for (r, c), pid in occupied.items():
        result[r, c] = pid
    return result, total_turns


def test(N, players, label):
    print(f"\n{'='*60}")
    print(f"Test: {label} (N={N})")
    print(f"{'='*60}")
    
    seq = generate_inverted_spiral(N)
    ref_occ, ref_turns = brute_force_simulate(N, seq, players)
    
    game = Game(N, seq, players)
    opt_turns = game.run()
    opt_occ = game.occupied
    
    match = np.array_equal(ref_occ, opt_occ)
    print(f"  Reference: {ref_turns} turns")
    print(f"  Optimized: {opt_turns} turns")
    print(f"  Match: {match}")
    
    if not match:
        print("  *** MISMATCH ***")
        if N <= 10:
            print(f"  Ref:\n{ref_occ}")
            print(f"  Opt:\n{opt_occ}")
            diff = np.argwhere(ref_occ != opt_occ)
            print(f"  {len(diff)} differing cells")
    else:
        print("  ✓ PASS")
    return match

ok = True
ok &= test(6, [{'color':'r','pieces':['Bishop']},{'color':'b','pieces':['Bishop']}], "Bishop vs Bishop")
ok &= test(6, [{'color':'r','pieces':['Rook']},{'color':'b','pieces':['Rook']}], "Rook vs Rook")
ok &= test(6, [{'color':'r','pieces':['Queen']},{'color':'b','pieces':['Queen']}], "Queen vs Queen")
ok &= test(8, [{'color':'r','pieces':['Knight','Bishop']},{'color':'b','pieces':['Rook']}], "Knight+Bishop vs Rook")
ok &= test(8, [{'color':'r','pieces':['Dragon']},{'color':'b','pieces':['Knight']}], "Dragon vs Knight")

print(f"\n{'='*60}")
print("ALL PASS ✓" if ok else "FAILURES DETECTED ✗")
print(f"{'='*60}")
