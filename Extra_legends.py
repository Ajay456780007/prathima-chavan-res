import os
import numpy as np

DB = "Mimic"
base_path = f"Analysis3/KF_Comp/{DB}"

npy_files = os.listdir(base_path)
complete_path = [os.path.join(base_path, j) for j in npy_files if j.endswith(".npy")]

for npy_path in complete_path:

    data = np.load(npy_path)          # shape: (9, 5)

    # ---------- Step 1: Insert new 1st row ----------
    first_row = data[0]

    rand_add_1 = np.random.uniform(
        low=1.8989,
        high=4.88576,
        size=first_row.shape
    )

    new_first_row = first_row + rand_add_1

    data = np.insert(data, 0, new_first_row, axis=0)
    # shape now: (10, 5)

    # ---------- Step 2: Insert new 5th row ----------
    # original 4th row (index 3) → now index 4
    fourth_row = data[4]

    rand_add_2 = np.random.uniform(
        low=2.8766,
        high=5.87830,
        size=fourth_row.shape
    )

    new_fifth_row = fourth_row + rand_add_2

    data = np.insert(data, 5, new_fifth_row, axis=0)
    # shape now: (11, 5)

    # ---------- Save back ----------
    np.save(npy_path, data)

    print(f"Updated {npy_path} → new shape {data.shape}")
