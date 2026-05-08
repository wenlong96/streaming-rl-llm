"""
Download chess sound effects for the Playground.

Uses Lichess's "piano" sound pack from the open-source `lila` repository
(https://github.com/lichess-org/lila), which is licensed AGPLv3+ and free
to redistribute. The script downloads the files we need and renames them
to the names the frontend expects, dropping them into a ./sounds/ folder.

Usage:
    python build_sounds.py
    python build_sounds.py --pack sfx       # use 'sfx' pack instead
    python build_sounds.py --out sounds     # output directory

Run once. To swap sound packs later, re-run with --pack and the new files
overwrite the old ones (no frontend change needed).

License: Lichess piano + sfx packs are AGPLv3+, redistributable for any use.
You should keep a NOTICE next to your sounds/ folder crediting Enigmahack and
linking https://github.com/lichess-org/lila — this script writes one for you.
"""

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Map of (frontend filename -> Lichess source filename in the pack).
# Some sounds reuse the same source — e.g. castling sounds the same as a capture
# in the piano pack since there's no dedicated castle sound. Promotion + game
# start both use GenericNotify.
DEFAULT_MAPPING = {
    "move.mp3":       "Move.mp3",
    "capture.mp3":    "Capture.mp3",
    "castle.mp3":     "Capture.mp3",       # piano pack has no castle sound; use capture
    "check.mp3":      "Check.mp3",
    "checkmate.mp3":  "Defeat.mp3",        # dramatic flourish
    "promotion.mp3":  "GenericNotify.mp3", # cleanish chime
    "game-start.mp3": "GenericNotify.mp3",
    "game-end.mp3":   "Draw.mp3",
    "illegal.mp3":    "Error.mp3",
}

# Some pack/file combos use .ogg, some .mp3. The piano and sfx packs use .mp3.
def base_url(pack):
    return f"https://raw.githubusercontent.com/lichess-org/lila/master/public/sound/{pack}"


def download(url, dest, ua):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="piano",
                   help="Lichess sound pack: piano, sfx, nes, instrument, lisp, wood (default: piano)")
    p.add_argument("--out", default="sounds", help="Output directory (default: sounds)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_url = base_url(args.pack)
    ua = "WiLLi-Playground-SoundFetch/1.0"

    print(f"Fetching from Lichess pack '{args.pack}' -> {pack_url}")
    print(f"Output directory: {out_dir.resolve()}\n")

    # Some packs use .mp3, some .ogg — we try both
    failed = []
    succeeded = 0
    for local_name, source_name in DEFAULT_MAPPING.items():
        dest = out_dir / local_name
        # Try the source extension first, then swap to .ogg if .mp3 fails
        candidates = [source_name]
        if source_name.endswith(".mp3"):
            candidates.append(source_name.replace(".mp3", ".ogg"))
        elif source_name.endswith(".ogg"):
            candidates.append(source_name.replace(".ogg", ".mp3"))

        ok = False
        for cand in candidates:
            url = f"{pack_url}/{cand}"
            try:
                size = download(url, dest, ua)
                # Rename if we got an .ogg for an .mp3-target
                if not local_name.endswith(cand.split(".")[-1]):
                    new_dest = out_dir / (local_name.rsplit(".",1)[0] + "." + cand.split(".")[-1])
                    dest.rename(new_dest)
                    dest = new_dest
                print(f"  ✓ {dest.name} ({size:,} bytes)")
                succeeded += 1
                ok = True
                break
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"  ! HTTP {e.code} fetching {cand}: {e}")
            except Exception as e:
                print(f"  ! Error fetching {cand}: {e}")
        if not ok:
            failed.append((local_name, candidates))
            print(f"  ✗ {local_name} — could not fetch any of: {candidates}")

    # Write the NOTICE file
    notice = f"""Sound effects in this directory are sourced from the Lichess `lila` repository:
    https://github.com/lichess-org/lila
Pack: {args.pack}
Author: Enigmahack
License: GNU Affero General Public License v3.0 or later (AGPLv3+)

Files were downloaded and renamed for use by the WiLLi Playground chess feature.
The original files are available in the lila repository under public/sound/{args.pack}/.
"""
    (out_dir / "NOTICE.txt").write_text(notice)

    print(f"\nDone. {succeeded}/{len(DEFAULT_MAPPING)} files downloaded.")
    if failed:
        print(f"\n{len(failed)} files failed. The frontend gracefully ignores missing sounds —")
        print("you can re-run with --pack sfx for an alternate set.")
    print(f"\nDrop the {out_dir}/ folder next to your index.html and the playground will pick it up.")


if __name__ == "__main__":
    main()
