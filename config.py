class Config:
    # --- Data ---
    DATA_ROOT    = "./dl_challenge"
    NUM_POINTS   = 2048        # Global scene point cloud size (after random subsampling)
    N_OBJ_POINTS = 256         # Points sampled per object for the per-object PC encoder
    IMG_SIZE     = (224, 224)
    MAX_OBJECTS  = 60
    TRAIN_RATIO  = 0.7
    VAL_RATIO    = 0.85        # Test is remaining 0.15
    AUGMENT      = False       # False = overfitting experiment; True = generalization training

    # --- Model ---
    # Global scene point cloud encoder — 4 layers
    POINT_HIDDEN_DIMS   = [64, 128, 256, 512]
    POINT_POST_DIM      = 512

    # Per-object point cloud encoder — no BatchNorm (avoids zero-slot stat contamination)
    OBJ_PC_HIDDEN_DIMS  = [64, 128, 256]
    OBJ_PC_OUTPUT_DIM   = 256

    # Image backbone: ResNet18 layer4 channels (frozen, masked avg-pool)
    IMG_OUTPUT_DIM      = 512

    # Per-object OBB decoder — 3 hidden layers → 12 raw params per slot
    #   center(3) + log_size(3) + rot6d(6) → _decode_obb() → 8 corners
    DECODER_HIDDEN_DIMS = [1024, 512, 256]
    DROPOUT_RATE        = 0.0

    # FUSED_DIM = global_pc + per_obj_pc + per_obj_img = 512 + 256 + 512 = 1280
    FUSED_DIM = POINT_POST_DIM + OBJ_PC_OUTPUT_DIM + IMG_OUTPUT_DIM

    # --- Training ---
    BATCH_SIZE    = 4
    EPOCHS        = 300
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY  = 0          # 0 for overfitting experiment; restore 1e-4 for generalization
    CLIP_GRAD     = 10.0
    FREEZE_BACKBONE     = True
    L1_WARMUP_EPOCHS    = 5

    # --- Optimization ---
    SCHEDULER_PATIENCE      = 20
    SCHEDULER_FACTOR        = 0.5
    EARLY_STOPPING_PATIENCE = 50

    # --- Loss ---
    L1_WEIGHT     = 1.0        # Weight for corner L1 term
    CENTER_WEIGHT = 0.5        # Weight for explicit center L1 term (after warmup)

    # --- Eval ---
    OBB_IOU_SAMPLES = 2048     # Monte Carlo samples for obb_3d_iou() evaluation
