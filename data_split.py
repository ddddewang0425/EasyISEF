import json
import random

def split_json_data(
    input_json_path: str,
    train_json_path: str,
    val_json_path: str,
    test_json_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    JSON 파일을 로드한 뒤, (train_ratio : val_ratio : test_ratio)로 데이터를 분할하여
    각각 별도의 JSON 파일로 저장합니다.
    
    Parameters
    ----------
    input_json_path : str
        원본 JSON 파일 경로
    train_json_path : str
        Train 용으로 분할된 JSON 파일 저장 경로
    val_json_path : str
        Validation 용으로 분할된 JSON 파일 저장 경로
    test_json_path : str
        Test 용으로 분할된 JSON 파일 저장 경로
    train_ratio : float
        Train 셋 비율 (기본값 0.8 → 80%)
    val_ratio : float
        Validation 셋 비율 (기본값 0.1 → 10%)
    test_ratio : float
        Test 셋 비율 (기본값 0.1 → 10%)
    random_seed : int
        분할 시 랜덤 시드 (기본값 42)
    """
    # 1. JSON 데이터 로드
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 랜덤 셔플 (재현성을 위해 random_seed 고정)
    random.seed(random_seed)
    random.shuffle(data)

    # 3. split size 계산
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # 남은 건 test로

    # 4. 실제 분할
    train_data = data[:train_size]
    val_data   = data[train_size:train_size + val_size]
    test_data  = data[train_size + val_size:]

    # 5. 각각 JSON으로 저장
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] total: {len(data)}")
    print(f"[INFO] train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

# 사용 예시:
if __name__ == "__main__":
    split_json_data(
        input_json_path="/home/gpuadmin/Desktop/RWKV/blip_laion_cc_sbu_558k.json",
        train_json_path="./data/train_blip_laion_cc_sbu_558k.json",
        val_json_path="./data/val_blip_laion_cc_sbu_558k.json",
        test_json_path="./data/test_blip_laion_cc_sbu_558k.json",
        train_ratio=0.96,
        val_ratio=0.02,
        test_ratio=0.02,
        random_seed=42
    )
