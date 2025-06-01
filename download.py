import vitaldb
import numpy as np

# Define the signals of interest
signals = ['PLETH', 'ART']

# Find case IDs that contain both PPG and ABP signals
case_ids = vitaldb.find_cases(signals)

# Loop through the case IDs and process each case
for cid in case_ids:
    try:
        # Load the case data at 500 Hz sampling rate
        data = vitaldb.load_case(cid, signals, 1/500)
        ppg = data[:, 0]
        abp = data[:, 1]

        # Save the data to a compressed NumPy file
        np.savez_compressed(f'case_{cid}.npz', ppg=ppg, abp=abp, fs=500)
    except Exception as e:
        print(f"Failed to process case {cid}: {e}")
