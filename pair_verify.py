import os

gt = sorted(os.listdir(r"Train\train\GT"))
lr = sorted(os.listdir(r"Train\train\NoisyLR"))

print("Same count:", len(gt)==len(lr))

mismatch = 0

for g,l in zip(gt,lr):
    if g != l:
        mismatch += 1

print("Mismatches:", mismatch)