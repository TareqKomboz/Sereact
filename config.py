import os

class Config:
    # --- Paths ---
    DATA_ROOT = "dl_challenge"
    
    # --- Architecture ---
    MAX_OBJECTS = 30
    NUM_POINTS  = 8192 # 2048
    POINT_HIDDEN_DIMS = [32, 64, 128, 256]
    POINT_POST_DIM    = 256
    
    OBJ_PC_HIDDEN_DIMS = [32, 64, 128, 256]
    OBJ_PC_OUTPUT_DIM  = 256
    N_OBJ_POINTS       = 512 # normalized points per object (doubled for detail)
    
    IMG_SIZE = (224, 224)
    IMG_FEAT_DIM = 2048 # ResNet50 Layer4 (ResNet18 was 512)
    
    # Fusion
    FUSED_DIM = POINT_POST_DIM + OBJ_PC_OUTPUT_DIM + IMG_FEAT_DIM # 1024+512+2048=3584
    DECODER_HIDDEN_DIMS = [2048, 512, 64]
    
    DROPOUT_RATE = 0.3
    
    # --- Dataset / Split ---
    TRAIN_RATIO = 0.7
    VAL_RATIO   = 0.85  # 0.8-0.9 is val, 0.9-1.0 is test
    AUGMENT     = True # Synchronized orientation + direct supervision
    
    # --- Training ---
    BATCH_SIZE    = 16         # Increased for better gradient stability with higher LR
    EPOCHS        = 10        # Reduced for fast super-convergence
    LEARNING_RATE = 1e-4       # Lowered for stable backbone fine-tuning
    WEIGHT_DECAY  = 1e-3       # Increased to counteract orientation overfitting
    CLIP_GRAD     = 10.0
    FREEZE_BACKBONE     = True

    # --- Optimization ---
    SCHEDULER_PATIENCE      = 10   # Note: OneCycleLR manages its own schedule
    SCHEDULER_FACTOR        = 0.5
    ORIENTATION_JITTER = 7.0      # Small "shimmy" on top of 90-deg steps
    EARLY_STOPPING_PATIENCE = 40

    # --- Loss Weights (Pure Component-Based) ---
    CENTER_WEIGHT = 1.0
    SIZE_WEIGHT   = 1.0
    ORIENT_WEIGHT = 1.0
    CONF_WEIGHT   = 1.0
    MASK_WEIGHT   = 1.0       # Higher weight for segmentation details

    # --- Instance Segmentation ---
    MASK_RESOLUTION = 28      # Standard Mask R-CNN resolution (28x28)

    # --- Eval & Viz ---
    OBB_IOU_SAMPLES = 2048     # Monte Carlo samples
    CONF_THRESHOLD  = 0.5      # Minimum confidence to visualize/evaluate
