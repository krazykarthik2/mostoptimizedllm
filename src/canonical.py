import numpy as np

# Workspace Bounds (Meters)
# Used for State Normalization
WORKSPACE_X = [0.2, 0.6]
WORKSPACE_Y = [-0.4, 0.4]
WORKSPACE_Z = [0.0, 0.5]

# Canonical State Scales (Half-ranges)
STATE_SCALES = np.array([0.2, 0.4, 0.25], dtype=np.float32)
STATE_OFFSETS = np.array([0.4, 0.0, 0.25], dtype=np.float32)

# Action Quantile Stats (Meters)
# Based on BridgeData V2 relative deltas (1st and 99th percentiles)
# Used for Action (Delta) Normalization
ACTION_Q01 = np.array([-0.0290, -0.0449, -0.0303, 0.0], dtype=np.float32)
ACTION_Q99 = np.array([0.0273, 0.0456, 0.0522, 1.0], dtype=np.float32)
ACTION_RANGE = ACTION_Q99 - ACTION_Q01

def normalize_state(pos_meters, gripper_raw):
    """Meters [3] + Gripper [1] -> Normalized [-1, 1] [4]"""
    norm_pos = (pos_meters - STATE_OFFSETS) / STATE_SCALES
    return np.array([norm_pos[0], norm_pos[1], norm_pos[2], gripper_raw], dtype=np.float32)

def denormalize_state(state_norm):
    """Normalized [-1, 1] [4] -> Meters [3] + Gripper [1]"""
    pos_phys = (state_norm[:3] * STATE_SCALES) + STATE_OFFSETS
    return pos_phys, state_norm[3]

def normalize_action(delta_meters, gripper_target):
    """Delta Meters [3] + Gripper [1] -> Normalized [-1, 1] [4]"""
    raw = np.array([delta_meters[0], delta_meters[1], delta_meters[2], gripper_target], dtype=np.float32)
    norm = 2.0 * (raw - ACTION_Q01) / ACTION_RANGE - 1.0
    return np.clip(norm, -1.0, 1.0)

def denormalize_action(action_norm):
    """Normalized [-1, 1] [4] -> Delta Meters [3] + Gripper [1]"""
    raw = (action_norm + 1.0) / 2.0 * ACTION_RANGE + ACTION_Q01
    return raw[:3], raw[3]
