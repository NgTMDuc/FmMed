import numpy as np

def safe_load_npy(file_path):
    """Try loading a .npy file safely. Return True if successful, False otherwise."""
    try:
        _ = np.load(file_path)
        return True  # Load thành công
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return False  # Load thất bại

def filter_valid_samples(input_file, output_file):
    valid_samples = []

    with open(input_file, "r") as f:
        file_paths = [line.strip() for line in f.readlines()]

    total_files = len(file_paths)
    print(f"📂 Total files to check: {total_files}")

    for path in file_paths:
        if safe_load_npy(path):  # Chỉ cần load được, không kiểm tra shape
            valid_samples.append(path)

    # Lưu lại danh sách file hợp lệ
    with open(output_file, "w") as f:
        f.writelines("\n".join(valid_samples) + "\n")

    print(f"✅ Successfully loaded: {len(valid_samples)}/{total_files} files")
    print(f"📄 Saved filtered list to {output_file}")

# Lọc file train.txt và lưu vào train_new.txt
filter_valid_samples("train.txt", "train_new.txt")
