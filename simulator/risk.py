import numpy as np


# ==========================================
# Ego vehicle dimensions
# ==========================================

EGO_HALF_WIDTH = 1.0
EGO_HALF_LENGTH = 2.0


# ==========================================
# Collision settings
# ==========================================

COLLISION_THRESHOLD = 1.05


# ==========================================
# Risk settings
# ==========================================

RISK_SCALE = 0.25


def score_risk(candidates, agents, times):
    """
    Score every candidate trajectory against
    every predicted agent.

    Returns:
        risks
        collisions
    """

    risks = []
    collisions = []

    # ==========================================
    # Check every candidate
    # ==========================================

    for candidate in candidates:

        trajectory = candidate["trajectory"]

        total_risk = 0.0
        collision_count = 0

        # ======================================
        # Check every agent
        # ======================================

        for agent in agents:

            # Predict this agent
            agent_positions, agent_sigma = predict_agent(
                agent,
                times
            )

            # ==================================
            # Check every timestep
            # ==================================

            for i in range(len(trajectory)):

                ego_position = trajectory[i]

                agent_position = agent_positions[i]

                # --------------------------------
                # Difference between vehicles
                # --------------------------------

                dx = ego_position[0] - agent_position[0]
                dy = ego_position[1] - agent_position[1]

                # --------------------------------
                # Ellipse dimensions
                # --------------------------------

                lateral_radius = (
                    EGO_HALF_WIDTH
                    + agent.radius
                )

                forward_radius = (
                    EGO_HALF_LENGTH
                    + agent.radius
                )

                # --------------------------------
                # Ellipse-normalised distance
                # --------------------------------

                dn = np.sqrt(
                    (dx / lateral_radius) ** 2
                    +
                    (dy / forward_radius) ** 2
                )

                # --------------------------------
                # Collision
                # --------------------------------

                if dn < COLLISION_THRESHOLD:

                    collision_count += 1

                # --------------------------------
                # Soft risk
                # --------------------------------

                uncertainty = agent_sigma[i]

                s = (
                    RISK_SCALE
                    + uncertainty /
                    max(lateral_radius, forward_radius)
                )

                risk = np.exp(
                    -0.5 *
                    ((dn - 1.0) / s) ** 2
                )

                total_risk += risk

        risks.append(total_risk)
        collisions.append(collision_count)

    return (
        np.array(risks),
        np.array(collisions)
    )


def predict_agent(agent, times):
    """
    Constant velocity prediction used by risk scoring.
    """

    position = np.asarray(
        agent.position,
        dtype=float
    )

    velocity = np.asarray(
        agent.velocity,
        dtype=float
    )

    speed = np.linalg.norm(velocity)

    # ------------------------------------------
    # Future positions
    # ------------------------------------------

    positions = (
        position
        +
        times[:, None] * velocity
    )

    # ------------------------------------------
    # Growing uncertainty
    # ------------------------------------------

    sigma = (
        0.2
        +
        0.1 * times
        +
        0.1 * speed * times
    )

    return positions, sigma