class Config:
    # --- Data ---
    DATA_ROOT = "./dl_challenge"
    NUM_POINTS = 2048
    IMG_SIZE = (224, 224)
    MAX_OBJECTS = 60
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.85 # Relative to total (Test is remaining 0.15)
    
    # --- Model ---
    POINT_HIDDEN_DIMS = [64, 128, 512]
    POINT_POST_DIM = 512
    IMG_OUTPUT_DIM = 512  # ResNet18 layer4 output channels (used for masked avg-pool)
    DECODER_HIDDEN_DIM = 512
    DROPOUT_RATE = 0.0
    FUSED_DIM = POINT_POST_DIM + IMG_OUTPUT_DIM  # 512 + 512 = 1024
    
    # --- Training ---
    BATCH_SIZE = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4  # Bug #7: was 3e-3, too high for AdamW with frozen backbone
    WEIGHT_DECAY = 1e-4   # Bug #7: was 0, enables L2 regularization to reduce overfitting
    CLIP_GRAD = 10.0
    FREEZE_BACKBONE = True
    L1_WARMUP_EPOCHS = 10  # Train with pure L1 before enabling DIoU
    
    # --- Optimization ---
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    EARLY_STOPPING_PATIENCE = 20
    
    # --- Loss ---
    L1_WEIGHT = 1.0
