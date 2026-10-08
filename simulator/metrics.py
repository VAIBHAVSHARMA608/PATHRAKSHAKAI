import numpy as np


def calculate_minimum_clearance(
    ego_history,
    agent_history,
    agent_radius=0.8,
    ego_radius=1.0
):
    minimum_distance = float("inf")

    for ego_position, agent_position in zip(
        ego_history,
        agent_history
    ):

        distance = np.linalg.norm(
            ego_position - agent_position
        )

        clearance = (
            distance
            - agent_radius
            - ego_radius
        )

        minimum_distance = min(
            minimum_distance,
            clearance
        )

    return minimum_distance


def calculate_rms_jerk(
    acceleration_history,
    dt
):
    acceleration_history = np.asarray(
        acceleration_history,
        dtype=float
    )

    if len(acceleration_history) < 2:
        return 0.0

    jerk = (
        np.diff(acceleration_history)
        / dt
    )

    rms_jerk = np.sqrt(
        np.mean(jerk ** 2)
    )

    return rms_jerk