import numpy as np

# Load the .npz file
data = np.load("clean_10s_strips_all.npz")

# List all arrays stored in the file
print("Keys inside the file:", data.files)

# Access and view each array
for key in data.files:
    arr = data[key]
    print(f"🔹 {key}: shape = {arr.shape}")
    # Check if the array is 0-dimensional
    if arr.ndim == 0:
        print(arr)
    else:
        print(arr[:10])  # show first 10 values