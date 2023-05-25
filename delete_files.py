import os
import glob

if __name__ == "__main__":
    file_pattern = "GAT/DP/**/*.adj"
    files = glob.glob(file_pattern, recursive=True)
    for file in files:
        file = file.replace('\\', '/')
        os.remove(file)
    pass
