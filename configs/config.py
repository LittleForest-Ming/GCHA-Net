# configs/config.py

class Config:
    # --- 硬件配置 ---
    DEVICE = 'cuda'

    # --- 模型输入尺寸 ---
    IMG_WIDTH = 640
    IMG_HEIGHT = 360

    # --- 锚点设置 ---
    NUM_QUERIES = 2700
    # 锚点生成范围 (虽然主要用于生成二次，但在三次模型中作为基准)
    N_K, N_M, N_B = 6, 15, 30
    K_RANGE = (-0.4, 0.4)
    M_RANGE = (-1.0, 1.0)
    B_RANGE = (0.0, 1.0)

    # --- 训练参数 ---
    # 如果之前是 1e-4，现在减小 10 倍
    LR_BACKBONE = 1e-5  # 甚至 1e-6
    LR_TRANSFORMER = 1e-4
    WEIGHT_DECAY = 1e-4

    MASK_THRESHOLD = 0.03
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 400
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 2

    # --- 路径 (请根据您的实际路径修改) ---
    DATA_ROOT = r"E:\PythonProject\crop protection\crop row detection\Dataset"