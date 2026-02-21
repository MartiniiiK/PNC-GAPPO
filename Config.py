import os
import torch

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PN_FILES_DIR = os.path.join(BASE_DIR, "PNfiles")

    TRAINING_RESULT_DIR = os.path.join(BASE_DIR, "Training_result")
    TRAINING_MODELS_DIR = os.path.join(TRAINING_RESULT_DIR, "Models")
    EVAL_RESULT_DIR = os.path.join(BASE_DIR, "Eval_result")
    RESULT_DIR = os.path.join(BASE_DIR, "Results")
    MODEL_DIR = os.path.join(BASE_DIR, "Models")
    BUFFER_SIZE = 1024

    GEN_MIN_JOB = 2
    GEN_MAX_JOB = 3
    GEN_MIN_OP = 4
    GEN_MAX_OP = 6
    GEN_MIN_RES = 3
    GEN_MAX_RES = 5
    GEN_MIN_TIME = 1
    GEN_MAX_TIME = 5
    GEN_MIN_LOT = 1
    GEN_MAX_LOT = 5
    GEN_RATIO_ALT = 0.1
    GEN_MIN_MULTI_R = 1
    GEN_MAX_MULTI_R = 1

    CURRICULUM_SCHEDULE = [
        {
            "name": "L1",
            "n_job_range": (3, 4), "n_op_range": (5, 6), "n_res_range": (6, 8),
            "lot_range": (1, 5), "ratio_alt": 0.5, "hot_res_ratio": 0.0,
            "hot_res_weight": 2.0, "branch_prob": 0.2, "branch_depth_range": (2, 2),
            "res_capacity_range": (1, 2), "continuous_hold": False, "upgrade_threshold": 0.90
        },
    ]
    CURRICULUM_WINDOW_SIZE = 100
    EVAL_INTERVAL = 50
    TRAIN_PN_LIST = []

    TRAIN_MODE = "random_gen"
    SINGLE_INSTANCE_PN = ""
    FIXED_PN_POOL = []

    GEN_POOL_SIZE = None
    GEN_POOL_MIN_SIZE = 30
    GEN_POOL_MAX_SIZE = 120
    GEN_POOL_REFRESH_INTERVAL = 200
    GEN_POOL_REFRESH_RATIO = 0.20
    GEN_POOL_TARGET_EPISODE_VISITS_PER_PN = 50
    GEN_POOL_TARGET_UNIQUE_PNS = 500
    GEN_POOL_EXPORT = False
    GEN_POOL_EXPORT_DIR = os.path.join(TRAINING_RESULT_DIR, "GenPoolPNs")

    FINAL_EVAL_ENABLED = True
    FINAL_EVAL_BENCHMARK_ENABLED = False
    FINAL_EVAL_RANDOM_PN_NUM = 20
    FINAL_EVAL_RANDOM_SEED = 123
    FINAL_EVAL_RANDOM_EXPORT = False
    FINAL_EVAL_RANDOM_EXPORT_DIR = os.path.join(TRAINING_RESULT_DIR, "FinalEval_RandomPNs")

    SAVE_TRAIN_HISTORY_PT = True
    SEED = 42
    LEARNING_RATE = 5e-5
    ROLLOUT_STEPS = 1024
    PPO_BATCH_SIZE = 128 * 5
    PPO_EPOCHS = 4
    CLIP_EPS_POLICY = 0.2
    CLIP_EPS_VALUE = 500.0 * 0.01
    TARGET_KL = 0.02
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENTROPY_COEF = 0.01
    ENTROPY_COEF_MIN = 0.01
    ENTROPY_ANNEAL_UPDATES = 0

    ADAPTIVE_ENTROPY = False
    ENTROPY_COEF_INIT = 0.01
    ENTROPY_LR = 1e-3
    TARGET_ENTROPY = 1.0

    VALUE_LOSS_COEF = 0.001
    MAX_GRAD_NORM = 0.5
    EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 500

    MAX_PLACES = 70
    MAX_TOKENS_PER_PLACE = 5
    MAX_TRANSITIONS = 80
    PAD_VALUE = -1.0

    DYNAMIC_STATE_DIM = MAX_PLACES + MAX_PLACES * MAX_TOKENS_PER_PLACE + MAX_PLACES
    CONTEXT_PLACE_DIM = MAX_PLACES * 3
    CONTEXT_STRUCT_DIM = 2 * (MAX_PLACES * MAX_TRANSITIONS)
    CONTEXT_DIM = CONTEXT_PLACE_DIM + CONTEXT_STRUCT_DIM

    AC_NET_TYPE = "mlp"
    AC_PLACE_EMB_DIM = 128
    AC_HIDDEN_DIM = 256
    AC_DROPOUT = 0.0

    GAT_NUM_LAYERS = 2
    GAT_NUM_HEADS = 4
    GAT_HEAD_DIM = 32
    GAT_FF_HIDDEN = 256
    GAT_DROPOUT = 0.0
    GAT_EDGE_BETA_INIT = 0.1

    REWARD_DEADLOCK = -500.0
    REWARD_SUCCESS_BONUS = 100.0
    REWARD_SCALE = 0.01

    PLOT_DPI = 300
    FONT_FAMILY = 'DejaVu Sans'
    FONT_SIZE_TITLE = 16
    FONT_SIZE_LABEL = 12
    FONT_SIZE_LEGEND = 10
    LINE_WIDTH = 2
