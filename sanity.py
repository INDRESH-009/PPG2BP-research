import numpy as np
d = np.load("4k/signals.npz")
ppg, abp = d["ppg"], d["abp"]

# How many NaNs?
print("NaNs in PPG:", np.isnan(ppg).sum(),
      "   NaNs in ABP:", np.isnan(abp).sum())

# Where does real data start?
first_good = np.where(~np.isnan(ppg))[0][0]
print("First non-NaN sample index:", first_good,
      "≈", round(first_good/500, 1), "s")
