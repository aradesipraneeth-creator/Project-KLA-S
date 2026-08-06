import os
import random
import numpy as np

folder = r"Train\train\NoisyLR"

files = random.sample(
    [f for f in os.listdir(folder) if f.endswith(".npy")],
    20
)

for f in files:
    img = np.load(os.path.join(folder,f))

    print(
        f,
        "Mean=", round(img.mean(),4),
        "Std=", round(img.std(),4)
    )