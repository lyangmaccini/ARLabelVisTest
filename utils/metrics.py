import numpy as np

def load_data(LAB_file, RGB_file):
    # Load LAB values
    lab_vals = []
    with open(LAB_file) as f:
        for line in f:
            l, a, b = line.strip().split(",")
            lab_vals.append([float(l), float(a), float(b)])

    lab_vals = np.array(lab_vals)

    # Build the 256x256x256x3 lookup array
    lookup = np.zeros((256, 256, 256, 3), dtype=np.float32)

    with open(RGB_file) as f:
        for i, line in enumerate(f):
            r, g, b = map(int, line.strip().split(","))
            lookup[r, g, b] = lab_vals[i]
    return lookup

