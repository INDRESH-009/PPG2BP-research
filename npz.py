import numpy as np
import sys

def view_npz(filepath):
    try:
        data = np.load(filepath)
        print(f"✅ Loaded: {filepath}")
        print(f"📦 Keys in .npz file: {list(data.keys())}")
        print("")

        for key in data:
            value = data[key]
            print(f"🔑 {key} → shape: {value.shape}, dtype: {value.dtype}")
            print(f"Sample values:\n{value}\n")

        data.close()
    except Exception as e:
        print(f"❌ Error reading .npz file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_npz.py <file.npz>")
    else:
        view_npz(sys.argv[1])
