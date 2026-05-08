"""
Build an opening book from your last N months of chess.com games.

Output: opening_book.json — a frequency-weighted map of FEN positions to UCI
moves, used by the Playground page to bias the bot's opening play toward your
real repertoire.

Usage:
    python build_opening_book.py
    python build_opening_book.py --username kwongkoonshing --months 1 --plies 8

Requires only the Python standard library + python-chess.
Install python-chess once with:    pip install chess

If python-chess is not available the script will tell you and exit cleanly.
"""

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import chess
    import chess.pgn
    import io
except ImportError:
    print("ERROR: python-chess is required. Install with:")
    print("    pip install chess")
    sys.exit(1)


def fetch_archives(username):
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    req = urllib.request.Request(url, headers={"User-Agent": "OpeningBookBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["archives"]


def fetch_games_for_month(archive_url):
    req = urllib.request.Request(archive_url, headers={"User-Agent": "OpeningBookBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["games"]


def position_key(board):
    """Use FEN piece placement + side-to-move only (strip move counters / castling /
    ep) so positions reached via different move orders match."""
    parts = board.fen().split()
    return f"{parts[0]} {parts[1]}"


def build_book(username, months_back, plies):
    print(f"Fetching archives for {username}...")
    archives = fetch_archives(username)
    if not archives:
        print("No archives found — is the username public?")
        return None

    # Take the most recent N archives
    targets = archives[-months_back:]
    print(f"Using last {len(targets)} month(s): {[a.split('/')[-2:] for a in targets]}")

    # Map: fen_key -> {uci_move: count}
    book = defaultdict(lambda: defaultdict(int))
    games_used = 0
    games_skipped = 0

    for arch in targets:
        print(f"  Fetching {arch}...")
        games = fetch_games_for_month(arch)
        for g in games:
            pgn_text = g.get("pgn", "")
            if not pgn_text:
                games_skipped += 1
                continue

            # Determine the player's color
            white = g.get("white", {}).get("username", "").lower()
            black = g.get("black", {}).get("username", "").lower()
            uname = username.lower()
            if uname == white:
                player_color = chess.WHITE
            elif uname == black:
                player_color = chess.BLACK
            else:
                games_skipped += 1
                continue

            try:
                game_obj = chess.pgn.read_game(io.StringIO(pgn_text))
            except Exception:
                games_skipped += 1
                continue
            if game_obj is None:
                games_skipped += 1
                continue

            board = game_obj.board()
            ply_count = 0
            for move in game_obj.mainline_moves():
                if ply_count >= plies * 2:  # plies counts full moves, *2 = half-moves
                    break
                # Only record the player's moves (the bot will be playing as them)
                if board.turn == player_color:
                    key = position_key(board)
                    book[key][move.uci()] += 1
                board.push(move)
                ply_count += 1
            games_used += 1

    print(f"\nProcessed: {games_used} games used, {games_skipped} skipped")
    print(f"Unique positions in book: {len(book)}")

    # Convert to output format
    positions = {}
    for fen_key, moves in book.items():
        # Filter: drop moves played only once at positions with multiple options
        # (singletons in popular positions are noise; keep singletons in singleton
        # positions because they're all you've ever played there)
        items = [(uci, cnt) for uci, cnt in moves.items()]
        items.sort(key=lambda x: -x[1])
        positions[fen_key] = [{"uci": uci, "weight": cnt} for uci, cnt in items]

    return {
        "_meta": {
            "source": "chess.com",
            "username": username,
            "months_back": months_back,
            "plies": plies,
            "games_used": games_used,
            "games_skipped": games_skipped,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "positions": positions,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--username", default="kwongkoonshing")
    p.add_argument("--months", type=int, default=1, help="How many months back to scan")
    p.add_argument("--plies", type=int, default=8, help="How many full moves of opening to record")
    p.add_argument("--out", default="opening_book.json")
    args = p.parse_args()

    book = build_book(args.username, args.months, args.plies)
    if book is None:
        sys.exit(1)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(book, indent=2))
    print(f"\nWrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Drop this file next to your index.html and the playground will pick it up.")


if __name__ == "__main__":
    main()
