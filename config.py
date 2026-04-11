import os

class Config:
    # --- Paths ---
    DATA_ROOT = "dl_challenge"
    
    # --- Architecture ---
    MAX_OBJECTS = 60
    NUM_POINTS  = 2048
    POINT_HIDDEN_DIMS = [64, 128, 256]
    POINT_POST_DIM    = 512
    
    OBJ_PC_HIDDEN_DIMS = [64, 128]
    OBJ_PC_OUTPUT_DIM  = 256
    N_OBJ_POINTS       = 256 # normalized points per object
    
    IMG_SIZE = (224, 224)
    IMG_FEAT_DIM = 512  # ResNet18 Layer4
    
    # Fusion
    FUSED_DIM = POINT_POST_DIM + OBJ_PC_OUTPUT_DIM + IMG_FEAT_DIM # 512+256+512=1280
    DECODER_HIDDEN_DIMS = [1024, 512, 256]
    
    DROPOUT_RATE = 0.1
    
    # --- Dataset / Split ---
    TRAIN_RATIO = 0.8
    VAL_RATIO   = 0.9  # 0.8-0.9 is val, 0.9-1.0 is test
    AUGMENT     = True # Synchronized rotation + direct supervision
    
    # --- Training ---
    BATCH_SIZE    = 16         # Increased for better gradient stability with higher LR
    EPOCHS        = 300        # Reduced for fast super-convergence
    LEARNING_RATE = 5e-3       # Peak LR for OneCycleLR
    WEIGHT_DECAY  = 1e-5       # Reduced slightly to help escape rotation local minima
    CLIP_GRAD     = 10.0
    FREEZE_BACKBONE     = True

    # --- Optimization ---
    SCHEDULER_PATIENCE      = 10   # Note: OneCycleLR manages its own schedule
    SCHEDULER_FACTOR        = 0.5
    EARLY_STOPPING_PATIENCE = 40

    # --- Loss Weights (Pure Component-Based) ---
    CENTER_WEIGHT = 1       # Restored to 1.0 for balanced translation
    SIZE_WEIGHT   = 1       # Significantly reduced as size is already stable/low
    ORIENT_WEIGHT = 1       # Doubled to 10.0 to force orientation progress
    CONF_WEIGHT   = 1       # Reduced as objectness is already nearly solved

    # --- Eval & Viz ---
    OBB_IOU_SAMPLES = 2048     # Monte Carlo samples
    CONF_THRESHOLD  = 0.5      # Minimum confidence to visualize/evaluate
