import numpy as np


# =========================
# Simulation settings
# =========================

DT = 0.1


# =========================
# Vehicle limits
# =========================

MAX_ACCELERATION = 2.0
MAX_DECELERATION = 4.0

MAX_LATERAL_ACCELERATION = 3.0


# =========================
# Lateral controller
# =========================

LATERAL_GAIN = 2.0
LATERAL_DAMPING = 2.0


def ego_step(state, target_y, target_v):
    """
    Move the ego vehicle forward by one simulation timestep.

    state:
        x  -> lateral position
        y  -> forward position
        vx -> lateral velocity
        vy -> forward velocity
        ax -> lateral acceleration
        ay -> forward acceleration

    target_y:
        Desired lateral position.

    target_v:
        Desired forward speed.

    Returns:
        Updated vehicle state.
    """

    # ==================================
    # 1. Current vehicle values
    # ==================================

    current_v = state["vy"]
    current_y = state["x"]
    current_vx = state["vx"]

    # ==================================
    # 2. Forward speed control
    # ==================================

    speed_error = target_v - current_v

    desired_acceleration = speed_error

    acceleration = np.clip(
        desired_acceleration,
        -MAX_DECELERATION,
        MAX_ACCELERATION
    )

    state["ay"] = acceleration

    # ==================================
    # 3. Lateral position control
    # ==================================

    lateral_error = target_y - current_y

    desired_lateral_acceleration = (
        LATERAL_GAIN * lateral_error
        - LATERAL_DAMPING * current_vx
    )

    lateral_acceleration = np.clip(
        desired_lateral_acceleration,
        -MAX_LATERAL_ACCELERATION,
        MAX_LATERAL_ACCELERATION
    )

    state["ax"] = lateral_acceleration

    # ==================================
    # 4. Update velocities
    # ==================================

    state["vy"] += state["ay"] * DT

    state["vx"] += state["ax"] * DT

    # Prevent reverse movement for now
    state["vy"] = max(state["vy"], 0.0)

    # ==================================
    # 5. Update positions
    # ==================================

    state["y"] += state["vy"] * DT

    state["x"] += state["vx"] * DT

    # ==================================
    # 6. Return updated state
    # ==================================

    return state