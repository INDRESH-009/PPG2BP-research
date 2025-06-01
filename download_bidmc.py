# src/scripts/download_bidmc.py
#!/usr/bin/env python3
"""
BIDMC PPG+ABP downloader and NPZ exporter.

Usage:
    python src/scripts/download_bidmc.py --records 105 107 --out data/raw/bidmc

Downloads specified BIDMC records from PhysioNet, extracts PLETH and ART channels, and saves each as:
    data/raw/bidmc/<record>.npz
        ├─ ppg : float32[n_samples]
        ├─ abp : float32[n_samples]
        └─ fs  : int16 (sampling rate, 100 Hz)
"""
import argparse
import pathlib
import wfdb
import numpy as np

def download_and_save(record: str, out_dir: pathlib.Path):
    # Create output directory for this record
    rec_dir = out_dir / record
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Download header and signal data
    # Download only this record by specifying records list
    wfdb.dl_database('bidmc', str(out_dir), records=[record])

    # Read full signal
    sig, meta = wfdb.rdsamp(str(out_dir / record))
    names = [n.upper() for n in meta['sig_name']]
    fs = meta['fs']  # sampling frequency

    # Extract PPG and ABP channels
    try:
        ppg = sig[:, names.index('PLETH')]
        abp = sig[:, names.index('ART')]
    except ValueError as e:
        print(f"[!] Record {record} missing required channels: {e}")
        return

    # Save as compressed NPZ
    out_file = rec_dir / f"{record}.npz"
    np.savez_compressed(
        out_file,
        ppg=ppg.astype('float32'),
        abp=abp.astype('float32'),
        fs=np.int16(fs)
    )
    print(f"✅ Saved {record}.npz ({len(ppg)} samples @ {fs} Hz)")


def main(records: list[str], out_root: pathlib.Path):
    out_root.mkdir(parents=True, exist_ok=True)
    for rec in records:
        download_and_save(rec, out_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download BIDMC PPG+ABP records.')
    parser.add_argument(
        '--records', nargs='+', required=True,
        help='BIDMC record numbers to download, e.g. 105 107'
    )
    parser.add_argument(
        '--out', required=True,
        help='Output root folder, e.g. data/raw/bidmc'
    )
    args = parser.parse_args()

    main(args.records, pathlib.Path(args.out).expanduser())
