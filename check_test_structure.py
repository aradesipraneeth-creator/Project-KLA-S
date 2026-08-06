import os

test_folder = r"Test_NoisyLR"

for root, dirs, files in os.walk(test_folder):
    print("\nFolder:", root)
    print("Subfolders:", dirs)
    print("Number of Files:", len(files))

    if len(files) > 0:
        print("First 5 Files:", files[:5])