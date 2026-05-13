"""
Comprehensive verification suite for the game engine.
Uses a pure-Python incremental reference implementation for medium boards,
and post-hoc validation for any size.
"""
import numpy as np
import time
from generate_chessboards import generate_inverted_spiral
from game import Game, _piece_offsets, _piece_slider_mask, ALL8_DIRS, SLIDER_MASKS

# ---------------------------------------------------------------------------
# Pure-Python reference implementation (incremental, supports sliders)
# ---------------------------------------------------------------------------
def reference_simulate(N, sequence, players_config):
    num_players = len(players_config)
    occupied = np.full((N, N), -1, dtype=np.int8)
    control = np.zeros((num_players, N, N), dtype=np.int32)
    piece_smask = np.zeros((N, N), dtype=np.int32)
    piece_owner = np.full((N, N), -1, dtype=np.int32)

    order = np.argsort(sequence, axis=None)
    s_rows = order // N
    s_cols = order % N
    total = N * N
    seq_idx = np.zeros((N, N), dtype=np.int64)
    for i in range(total):
        seq_idx[s_rows[i], s_cols[i]] = i

    # Precompute offsets and slider masks per piece
    piece_data = []
    has_sliders = False
    for p in players_config:
        pdata = []
        for pname in p['pieces']:
            offs = _piece_offsets(pname)
            sm = _piece_slider_mask(pname)
            if sm: has_sliders = True
            pdata.append((offs, sm))
        piece_data.append(pdata)

    search_indices = [0] * num_players
    active = [True] * num_players
    piece_indices = [0] * num_players
    total_turns = 0
    dirs8 = ALL8_DIRS

    while any(active):
        for pid in range(num_players):
            if not active[pid]:
                continue
            pidx = piece_indices[pid] % len(players_config[pid]['pieces'])
            offs, smask = piece_data[pid][pidx]

            idx = search_indices[pid]
            found = False
            while idx < total:
                r, c = int(s_rows[idx]), int(s_cols[idx])
                if occupied[r, c] == -1:
                    dominated = False
                    for p in range(num_players):
                        if p != pid and control[p, r, c] > 0:
                            dominated = True
                            break
                    if not dominated:
                        occupied[r, c] = pid
                        piece_owner[r, c] = pid
                        piece_smask[r, c] = smask

                        # Ray blocking
                        if has_sliders:
                            for d in range(8):
                                dr, dc = int(dirs8[d, 0]), int(dirs8[d, 1])
                                bit = 1 << d
                                nr, nc = r - dr, c - dc
                                while 0 <= nr < N and 0 <= nc < N:
                                    if occupied[nr, nc] != -1:
                                        ow = int(piece_owner[nr, nc])
                                        pm = int(piece_smask[nr, nc])
                                        if pm & bit:
                                            br, bc = r + dr, c + dc
                                            while 0 <= br < N and 0 <= bc < N:
                                                control[ow, br, bc] -= 1
                                                fi = int(seq_idx[br, bc])
                                                for pp in range(num_players):
                                                    if pp != ow and search_indices[pp] > fi:
                                                        search_indices[pp] = fi
                                                if occupied[br, bc] != -1:
                                                    break
                                                br += dr
                                                bc += dc
                                        break
                                    nr -= dr
                                    nc -= dc

                        # Add leaps
                        for i in range(len(offs)):
                            nr = r + int(offs[i, 0])
                            nc = c + int(offs[i, 1])
                            if 0 <= nr < N and 0 <= nc < N:
                                control[pid, nr, nc] += 1

                        # Add rays
                        if smask:
                            for d in range(8):
                                if smask & (1 << d):
                                    dr, dc = int(dirs8[d, 0]), int(dirs8[d, 1])
                                    nr, nc = r + dr, c + dc
                                    while 0 <= nr < N and 0 <= nc < N:
                                        control[pid, nr, nc] += 1
                                        if occupied[nr, nc] != -1:
                                            break
                                        nr += dr
                                        nc += dc

                        total_turns += 1
                        piece_indices[pid] += 1
                        search_indices[pid] = idx + 1
                        found = True
                        break
                idx += 1
            if not found:
                search_indices[pid] = idx
                active[pid] = False

    return occupied, total_turns

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_test(N, players, label):
    seq = generate_inverted_spiral(N)

    t0 = time.time()
    ref_occ, ref_turns = reference_simulate(N, seq, players)
    ref_time = time.time() - t0

    game = Game(N, seq, players)
    t0 = time.time()
    opt_turns = game.run()
    opt_time = time.time() - t0
    opt_occ = game.occupied

    match = np.array_equal(ref_occ, opt_occ)
    turns_ok = ref_turns == opt_turns
    ok = match and turns_ok

    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {status} | {label} ({N}x{N}) | turns: {ref_turns:,} | "
          f"ref={ref_time:.2f}s opt={opt_time:.3f}s")

    if not ok:
        diff_count = int(np.sum(ref_occ != opt_occ))
        print(f"         Ref turns={ref_turns}, Opt turns={opt_turns}, "
              f"diff cells={diff_count}")
        if N <= 12:
            print(f"         Ref:\n{ref_occ}")
            print(f"         Opt:\n{opt_occ}")
    return ok

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Warmup numba
    seq = generate_inverted_spiral(8)
    Game(8, seq, [{'color':'r','pieces':['Knight']},{'color':'b','pieces':['Knight']}]).run()

    print("=" * 80)
    print("SMALL BOARD TESTS (brute-force verifiable)")
    print("=" * 80)
    all_ok = True
    all_ok &= run_test(4, [{'color':'r','pieces':['Knight']},{'color':'b','pieces':['Knight']}], "Knight vs Knight")
    all_ok &= run_test(8, [{'color':'r','pieces':['Knight']},{'color':'b','pieces':['Knight']}], "Knight vs Knight")
    all_ok &= run_test(8, [{'color':'r','pieces':['Rook']},{'color':'b','pieces':['Rook']}], "Rook vs Rook")
    all_ok &= run_test(8, [{'color':'r','pieces':['Queen']},{'color':'b','pieces':['Queen']}], "Queen vs Queen")
    all_ok &= run_test(8, [{'color':'r','pieces':['Dragon']},{'color':'b','pieces':['Knight']}], "Dragon vs Knight")
    all_ok &= run_test(10, [{'color':'r','pieces':['Bishop','Knight']},{'color':'b','pieces':['Rook']}], "Bishop+Knight vs Rook")
    all_ok &= run_test(10, [{'color':'r','pieces':['Knight']},{'color':'b','pieces':['Camel']},{'color':'g','pieces':['Zebra']}], "3-player leapers")
    all_ok &= run_test(10, [{'color':'r','pieces':['leaper_3_5']},{'color':'b','pieces':['Knight']}], "leaper_3_5 vs Knight")
    all_ok &= run_test(12, [{'color':'r','pieces':['Amazon','Giraffe','Giraffe','Giraffe']},{'color':'b','pieces':['Bishop','Knight']}], "Amazon+Giraffe vs Bishop+Knight")

    print()
    print("=" * 80)
    print("MEDIUM BOARD TESTS (500x500)")
    print("=" * 80)
    all_ok &= run_test(500, [{'color':'r','pieces':['Knight']},{'color':'b','pieces':['Knight']}], "Knight vs Knight")
    all_ok &= run_test(500, [{'color':'r','pieces':['Amazon','Giraffe','Giraffe','Giraffe']},{'color':'b','pieces':['Bishop','Knight']}], "Amazon+Giraffe vs Bishop+Knight")

    print()
    print("=" * 80)
    if all_ok:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)
