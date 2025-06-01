import numpy as np
import os
from scipy.signal import butter, filtfilt, decimate

def interpolate_nans_float32(signal: np.ndarray) -> np.ndarray:
    """
    Replace NaNs by linear interpolation (works in float32 to keep memory low).
    """
    sig = signal.astype(np.float32)
    nans = np.isnan(sig)
    not_nans = ~nans
    if np.all(nans):
        return sig  # cannot interpolate if all points are NaN

    idx = np.arange(len(sig))
    # np.interp will operate in float64 internally; cast result back to float32
    filled = np.interp(idx[nans], idx[not_nans], sig[not_nans]).astype(np.float32)
    sig[nans] = filled
    return sig

def butter_bandpass_filter(x: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 8.0, order: int = 2) -> np.ndarray:
    """
    Zero‐phase 2nd‐order Butterworth bandpass between lowcut and highcut (Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)

def compute_window_snr(ppg_window: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 8.0, order: int = 2) -> float:
    """
    For a 1D PPG window, bandpass‐filter it, treat that as 'signal',
    and the residual (window - filtered) as 'noise'. Then compute 10*log10(Var(signal)/Var(noise)).
    """
    filtered = butter_bandpass_filter(ppg_window, fs, lowcut, highcut, order)
    noise = ppg_window - filtered
    sig_pow = np.nanvar(filtered)
    noise_pow = np.nanvar(noise)
    if noise_pow <= 0:
        return np.inf
    return 10.0 * np.log10(sig_pow / noise_pow)

def process_one_npz(path: str,
                    target_fs: int = 50,
                    window_secs: int = 10,
                    overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a single .npz file (with 'ppg', 'abp', 'fs' arrays), return two arrays:

      - ppg_segments: shape (<=100, 500)   # 10 s @ 50 Hz
      - abp_segments: shape (<=100, 500)

    Steps:
     1) Load PPG/ABP/FS and cast to float32.
     2) Interpolate any NaNs (linear).
     3) Bandpass PPG (0.5–8 Hz) at original fs.
     4) Decimate both PPG & ABP to 50 Hz.
     5) Slide 10 s windows with 50% overlap; compute SNR on each PPG window.
     6) Sort windows by SNR, take top 100.
     7) Return the raw (downsampled) PPG/ABP slices for those top windows.
    """
    data = np.load(path)
    raw_ppg = data['ppg']    # usually float32 or float64
    raw_abp = data['abp']
    fs = float(data['fs'])   # e.g. 500.0

    # 1) Interpolate NaNs in float32
    ppg_interp = interpolate_nans_float32(raw_ppg)
    abp_interp = interpolate_nans_float32(raw_abp)

    # 2) Bandpass-filter PPG at original fs to remove baseline drift / high-frequency noise
    ppg_filtered = butter_bandpass_filter(ppg_interp, fs, lowcut=0.5, highcut=8.0, order=2).astype(np.float32)

    # 3) Downsample to target_fs (e.g. 50 Hz). We use scipy.signal.decimate with iir filter.
    decim_factor = int(fs // target_fs)
    if decim_factor < 1:
        raise ValueError(f"Original fs={fs} < target_fs={target_fs}.")
    ppg_ds = decimate(ppg_filtered, decim_factor, ftype='iir', zero_phase=True).astype(np.float32)
    abp_ds = decimate(abp_interp, decim_factor, ftype='iir', zero_phase=True).astype(np.float32)

    # 4) Sliding windows at 50 Hz: 10 s → 500 samples; overlap 50% → step = 250 samples
    win_len = window_secs * target_fs      # e.g. 10 * 50 = 500
    step = int(win_len * (1 - overlap))    # e.g. 500 * 0.5 = 250

    snr_list = []
    for start in range(0, len(ppg_ds) - win_len + 1, step):
        end = start + win_len
        win_ppg = ppg_ds[start:end]

        # 5) Compute SNR on this 10 s window
        snr_val = compute_window_snr(win_ppg, target_fs, lowcut=0.5, highcut=8.0, order=2)
        snr_list.append((snr_val, start, end))

    if len(snr_list) < 100:
        print(f"Warning: only {len(snr_list)} windows available in {os.path.basename(path)}.")

    # 6) Sort by descending SNR, pick top 100
    snr_list.sort(key=lambda x: x[0], reverse=True)
    top100 = snr_list[:100]

    # 7) Extract those slices from ppg_ds & abp_ds
    ppg_segs = np.zeros((len(top100), win_len), dtype=np.float32)
    abp_segs = np.zeros((len(top100), win_len), dtype=np.float32)

    for idx, (snr_val, start, end) in enumerate(top100):
        ppg_segs[idx, :] = ppg_ds[start:end]
        abp_segs[idx, :] = abp_ds[start:end]

    return ppg_segs, abp_segs

if __name__ == "__main__":
    # List all your uploaded files here:
    file_list = [
        "/Users/indreshmr/Desktop/javascript/case_3.npz"
    ]

    results = {}
    for fname in file_list:
        if not os.path.isfile(fname):
            print(f"File not found: {fname}")
            continue

        print(f"Processing {fname} ...")
        ppg_segments, abp_segments = process_one_npz(fname,
                                                     target_fs=50,
                                                     window_secs=10,
                                                     overlap=0.5)
        # Save or store however you like; here we just keep them in memory:
        results[fname] = {
            "ppg_10s_50Hz": ppg_segments,   # shape (<=100, 500)
            "abp_10s_50Hz": abp_segments    # shape (<=100, 500)
        }
        print(f" → Extracted {ppg_segments.shape[0]:3d} strips (each 10 s @ 50 Hz).")

    # Example: if you want to save them to disk, you could do:
    np.savez("clean_10s_strips_all.npz",
        **{f"{k}_ppg": v["ppg_10s_50Hz"] for k, v in results.items()},
        **{f"{k}_abp": v["abp_10s_50Hz"] for k, v in results.items()})
