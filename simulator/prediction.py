import numpy as np


# =========================
# Uncertainty parameters
# =========================

BASE_SIGMA = 0.2
TIME_SIGMA = 0.1
SPEED_SIGMA = 0.1


def predict(agent, times):
    """
    Predict the future position of an agent.

    Parameters:
        agent:
            Agent object containing:
                position
                velocity

        times:
            NumPy array containing future prediction times.

    Returns:
        positions:
            Predicted [x, y] positions.

        sigma:
            Uncertainty radius for each prediction.
    """

    # Current position
    position = np.asarray(agent.position, dtype=float)

    # Current velocity
    velocity = np.asarray(agent.velocity, dtype=float)

    # Speed magnitude
    speed = np.linalg.norm(velocity)

    # ==================================
    # Constant-velocity prediction
    # ==================================

    positions = position + times[:, None] * velocity

    # ==================================
    # Growing uncertainty
    # ==================================

    sigma = (
        BASE_SIGMA
        + TIME_SIGMA * times
        + SPEED_SIGMA * speed * times
    )

    return positions, sigma