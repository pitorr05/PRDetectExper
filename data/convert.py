import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# --- Cấu hình ---
BASE_DIR = "."  # Thư mục chứa script
DATA_DIR = os.path.join(BASE_DIR, "data")

# Đường dẫn file đầu vào
HC3_TRAIN_CSV = os.path.join(DATA_DIR, "HC3_train.csv")
HC3_TEST_CSV = os.path.join(DATA_DIR, "HC3_test_1k.csv")

# Đường dẫn file đầu ra (định dạng PRDetect)
OUTPUT_TRAIN = os.path.join(DATA_DIR, "hc3_train.jsonl")
OUTPUT_VAL = os.path.join(DATA_DIR, "hc3_val.jsonl")
OUTPUT_TEST = os.path.join(DATA_DIR, "hc3_test.jsonl")

# Số lượng mẫu mong muốn (lấy theo DetectRL)
TRAIN_SIZE = 7167
VAL_SIZE = 797
# TEST_SIZE = 1000 (lấy toàn bộ file test đã có)

# --- Hàm chuyển đổi DataFrame sang JSONL ---
def df_to_jsonl(df, output_path):
    """Ghi DataFrame vào file JSONL với cấu trúc text, label."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Đảm bảo text là string và label là int
            record = {
                "text": str(row['text']).strip(),
                "label": int(row['label'])
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✅ Đã tạo: {output_path} với {len(df)} mẫu")

# --- 1. Xử lý tập TRAIN và VAL từ HC3_train.csv ---
print(f"📖 Đang đọc file: {HC3_TRAIN_CSV}")
try:
    df_train_full = pd.read_csv(HC3_TRAIN_CSV)
    print(f"   Tổng số mẫu trong HC3_train.csv: {len(df_train_full)}")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file {HC3_TRAIN_CSV}. Vui lòng kiểm tra đường dẫn.")
    exit()

# Kiểm tra phân bố nhãn
print(f"   Nhãn 0 (Human): {len(df_train_full[df_train_full['label']==0])}")
print(f"   Nhãn 1 (AI): {len(df_train_full[df_train_full['label']==1])}")

# Lấy mẫu ngẫu nhiên để có số lượng chính xác cho train + val
# Tổng số mẫu cần lấy từ train là TRAIN_SIZE + VAL_SIZE = 7167 + 797 = 7964
total_needed = TRAIN_SIZE + VAL_SIZE
if len(df_train_full) >= total_needed:
    df_sampled = df_train_full.sample(n=total_needed, random_state=42)
    print(f"   Đã lấy mẫu {total_needed} mẫu từ file train.")
else:
    print(f"   ⚠️ File train chỉ có {len(df_train_full)} mẫu, ít hơn yêu cầu ({total_needed}). Sẽ dùng toàn bộ.")
    df_sampled = df_train_full

# Chia thành train và validation
df_train, df_val = train_test_split(
    df_sampled,
    test_size=VAL_SIZE,
    random_state=42,
    stratify=df_sampled['label']  # Giữ tỷ lệ nhãn
)

# --- 2. Xử lý tập TEST từ HC3_test_1k.csv ---
print(f"\n📖 Đang đọc file: {HC3_TEST_CSV}")
try:
    df_test = pd.read_csv(HC3_TEST_CSV)
    print(f"   Tổng số mẫu trong HC3_test_1k.csv: {len(df_test)}")
    # Đảm bảo chỉ lấy đúng 1000 mẫu (nếu file có hơn)
    if len(df_test) > 1000:
        df_test = df_test.sample(n=1000, random_state=42)
        print(f"   Đã lấy mẫu 1000 mẫu để đảm bảo số lượng.")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file {HC3_TEST_CSV}. Vui lòng kiểm tra đường dẫn.")
    exit()

# --- 3. Ghi các file JSONL ---
print("\n--- Bắt đầu ghi file ---")
df_to_jsonl(df_train, OUTPUT_TRAIN)
df_to_jsonl(df_val, OUTPUT_VAL)
df_to_jsonl(df_test, OUTPUT_TEST)

print("\n✅ Hoàn tất chuyển đổi dữ liệu HC3 sang định dạng PRDetect!")
print(f"📁 Các file được lưu trong thư mục: {DATA_DIR}")