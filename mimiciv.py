import wfdb, numpy as np, pathlib

ROOT = pathlib.Path("data/raw/mimiciv")
ROOT.mkdir(parents=True, exist_ok=True)

# Example subject (these IDs exist in every account)
record = "mimic4wdb/67/678/67890138/67890138_0001"

# Download header + dat
wfdb.dl_database(record, str(ROOT))

# Load
sig, fields = wfdb.rdsamp(str(ROOT/record))
ch = fields["sig_name"]
ppg = sig[:, ch.index("PLETH")]
abp = sig[:, ch.index("ABP")]

print("loaded", len(ppg), "samples @", fields["fs"], "Hz")
np.savez_compressed(ROOT/"67890138_0001.npz",
                    ppg=ppg.astype("float32"),
                    abp=abp.astype("float32"),
                    fs=np.int16(fields["fs"]))
