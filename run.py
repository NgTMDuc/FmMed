import os

def get_npy_files(root_path):
    """Duyệt qua toàn bộ thư mục và lấy danh sách các file .npy"""
    npy_files = []
    
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    
    return npy_files

# Sử dụng
root_path = "your/root/path"
npy_files = get_npy_files(root_path)
print(f"Found {len(npy_files)} .npy files.")

# Nếu muốn lưu vào file txt
with open("full_data.txt", "w") as f:
    f.writelines("\n".join(npy_files) + "\n")

print("✅ File list saved to full_data.txt")
