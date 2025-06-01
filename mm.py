#!/usr/bin/env python3
"""
VitalDB downloader with live probe counter.

    python -m src.scripts.download_vitaldb --num 10 --out data/raw
"""
from __future__ import annotations
import argparse, pathlib, sys, concurrent.futures as cf

import numpy as np
import tqdm, vitaldb

SIGS          = ["PLETH", "ART"]
MIN_SECONDS   = 300
MIN_SAMPLES   = MIN_SECONDS * 500
THREADS       = 4
START_ID      = 3_200          # skip short early shells
PROBE_PRINT   = 10             # show a probe message every N IDs


# ---------- helpers --------------------------------------------------------- #
def _matrix(ret):
    return ret[0] if isinstance(ret, tuple) else ret


def long_enough(cid: int) -> bool:
    try:
        probe = _matrix(vitaldb.load_case(cid, SIGS, 1))  # 1-Hz preview
        return probe.ndim == 2 and probe.shape[1] == 2 and len(probe) >= MIN_SECONDS
    except Exception:
        return False


def fetch_full(cid: int, root: pathlib.Path) -> bool:
    try:
        mat = _matrix(vitaldb.load_case(cid, SIGS))       # full 500 Hz
        if mat.ndim != 2 or mat.shape[1] != 2:
            return False
        ppg, abp = mat[:, 0], mat[:, 1]
        if len(ppg) < MIN_SAMPLES:
            return False

        case_dir = root / str(cid)
        case_dir.mkdir(exist_ok=True)
        np.savez_compressed(case_dir / "signals.npz",
                            ppg=ppg.astype("float32"),
                            abp=abp.astype("float32"),
                            fs=np.int16(500))
        return True
    except Exception as e:
        print(f"[!] case {cid} skipped: {e}", file=sys.stderr)
        return False


# ---------- main ------------------------------------------------------------ #
def main(n_cases: int, out_root: pathlib.Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    ids = vitaldb.find_cases(SIGS)[START_ID:]

    collected = 0
    probed = 0
    bar = tqdm.tqdm(total=n_cases, desc="downloaded")

    with cf.ThreadPoolExecutor(max_workers=THREADS) as pool:
        for cid in ids:
            if collected >= n_cases:
                break

            # 1-Hz probe
            probed += 1
            if probed % PROBE_PRINT == 0:
                tqdm.tqdm.write(f"probed {probed} IDs … collected {collected}")

            if not long_enough(cid):
                continue

            # full download (blocking—only one at a time keeps memory low)
            if fetch_full(cid, out_root):
                collected += 1
                bar.update(1)

    bar.close()
    print(f"✅  collected {collected} usable case(s) into {out_root}")


# ---------- CLI ------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=10,
                    help="number of long cases to fetch (default 10)")
    ap.add_argument("--out", required=True,
                    help="destination folder, e.g. data/raw")
    args = ap.parse_args()

    main(args.num, pathlib.Path(args.out).expanduser())
