#!/usr/bin/env python3
"""
Convert the BIDMC PPG+ABP .dat/.hea files into .npz for preprocessing.

Looks for:
    data/raw/bidmc/bidmcXX.dat
    data/raw/bidmc/bidmcXX.hea

and creates:
    data/raw/bidmc/bidmcXX/bidmcXX.npz
      ├─ ppg : float32[n_samples]
      ├─ abp : float32[n_samples]
      └─ fs  : int16

Usage:
    python src/scripts/convert_bidmc.py --in  data/raw/bidmc --out data/raw/bidmc_npz
"""
import argparse, pathlib, wfdb, numpy as np

def convert_record(hea_path: pathlib.Path, out_root: pathlib.Path):
    rec = hea_path.stem            # e.g. "bidmc01"
    dat_path = hea_path.with_suffix(".dat")
    if not dat_path.exists():
        print(f"[!] Missing .dat for {rec}, skipping")
        return

    # Read the signals
    sig, meta = wfdb.rdsamp(str(hea_path.with_suffix("")))
    names = [n.upper() for n in meta["sig_name"]]
    fs    = meta["fs"]

    # Extract PLETH & ART channels
    try:
        ppg = sig[:, names.index("PLETH")]
        abp = sig[:, names.index("ART")]
    except ValueError:
        print(f"[!] {rec}: required channels not found ({names})")
        return

    # Save into its own folder
    out_dir = out_root / rec
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{rec}.npz",
        ppg=ppg.astype("float32"),
        abp=abp.astype("float32"),
        fs =np.int16(fs),
    )
    print(f"✅ Converted {rec}: {len(ppg)} samples @ {fs} Hz")

def main(in_root: pathlib.Path, out_root: pathlib.Path):
    # Find all bidmcXX.hea files (exclude the 'n' respiration ones)
    hea_files = sorted(f for f in in_root.glob("bidmc*.hea")
                       if not f.name.endswith("n.hea"))
    out_root.mkdir(parents=True, exist_ok=True)
    for hea in hea_files:
        convert_record(hea, out_root)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="inroot",  required=True,
                   help="Root folder of BIDMC .hea/.dat files")
    p.add_argument("--out", dest="outroot", required=True,
                   help="Where to save per-record .npz folders")
    args = p.parse_args()
    main(pathlib.Path(args.inroot), pathlib.Path(args.outroot))
