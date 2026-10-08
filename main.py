import os
import argparse
import numpy as np
import copy

import matplotlib

# Prefer a live GUI backend on desktop Windows/macOS/Linux systems.
# Fall back to Agg only when a display is unavailable.
backend_candidates = [
    "TkAgg",
    "Qt5Agg",
    "QtAgg",
    "WXAgg",
    "MacOSX",
    "Agg",
]

if os.name == "nt":
    preferred = ["TkAgg", "Qt5Agg", "QtAgg", "Agg"]
else:
    preferred = ["TkAgg", "Qt5Agg", "QtAgg", "MacOSX", "Agg"]

backend = None
for candidate in preferred:
    try:
        matplotlib.use(candidate, force=True)
        backend = candidate
        break
    except Exception:
        continue

if backend is None:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

from simulator.ego import ego_step
from simulator.agents import Agent
from simulator.risk import score_risk


# ============================================================
# CONFIGURATION
# ============================================================

DT = 0.1
SIMULATION_STEPS = 60

EGO_RADIUS = 1.0
COW_RADIUS = 0.8

ROAD_WIDTH = 10
ROAD_LENGTH = 35


# ============================================================
# CANDIDATE TRAJECTORIES
# ============================================================

def rollout_candidates(initial_state):

    target_y_values = np.linspace(
        -4,
        4,
        9
    )

    target_speed_values = np.linspace(
        4,
        10,
        11
    )

    candidates = []

    for target_y in target_y_values:

        for target_speed in target_speed_values:

            state = copy.deepcopy(
                initial_state
            )

            trajectory = []

            for step in range(50):

                state = ego_step(
                    state,
                    target_y,
                    target_speed
                )

                trajectory.append([
                    state["x"],
                    state["y"]
                ])

            candidates.append({

                "target_y": target_y,

                "target_speed": target_speed,

                "trajectory": np.array(
                    trajectory
                )
            })

    return candidates


# ============================================================
# PATHRAKSHAK PLANNER
# ============================================================

def plan(state, agents):

    candidates = rollout_candidates(
        state
    )

    times = (
        np.arange(50)
        * DT
    )

    risks, collisions = score_risk(
        candidates,
        agents,
        times
    )

    best_index = None
    best_cost = float("inf")

    for i, candidate in enumerate(
        candidates
    ):

        risk_cost = risks[i]

        collision_cost = (
            collisions[i]
            * 1000
        )

        progress_reward = (
            candidate["trajectory"][-1, 1]
        )

        lane_cost = (
            abs(
                candidate["target_y"]
            )
            * 0.1
        )

        total_cost = (
            risk_cost
            + collision_cost
            - progress_reward
            + lane_cost
        )

        if total_cost < best_cost:

            best_cost = total_cost
            best_index = i

    return (
        candidates[best_index],
        candidates,
        risks,
        collisions
    )


# ============================================================
# BASELINE
# ============================================================

def baseline_plan(
    state,
    agents
):

    target_y = 0.0
    target_speed = 5.0

    for agent in agents:

        lateral_distance = abs(
            agent.position[0]
            - state["x"]
        )

        forward_distance = (
            agent.position[1]
            - state["y"]
        )

        if (
            lateral_distance < 1.5
            and
            0 < forward_distance < 12
        ):

            target_speed = 0.0

    return (
        target_y,
        target_speed
    )


# ============================================================
# SCENARIO
# ============================================================

def create_scenario():

    ego = {

        "x": 0.0,
        "y": 0.0,

        "vx": 0.0,
        "vy": 5.0,

        "ax": 0.0,
        "ay": 0.0
    }

    cow = Agent(

        position=[
            0.0,
            15.0
        ],

        velocity=[
            1.0,
            0.0
        ],

        radius=COW_RADIUS
    )

    return ego, [cow]


# ============================================================
# BASELINE SIMULATION
# ============================================================

def run_baseline():

    ego, agents = create_scenario()

    ego_history = []
    cow_history = []

    for step in range(
        SIMULATION_STEPS
    ):

        target_y, target_speed = (
            baseline_plan(
                ego,
                agents
            )
        )

        ego = ego_step(
            ego,
            target_y,
            target_speed
        )

        for agent in agents:

            agent.update(DT)

        ego_history.append([
            ego["x"],
            ego["y"]
        ])

        cow_history.append(
            agents[0].position.copy()
        )

    return (
        np.array(ego_history),
        np.array(cow_history)
    )


# ============================================================
# PATHRAKSHAK SIMULATION
# ============================================================

def run_pathrakshak():

    ego, agents = create_scenario()

    ego_history = []
    cow_history = []

    selected_paths = []
    risk_history = []
    collision_history = []

    for step in range(
        SIMULATION_STEPS
    ):

        (
            best_candidate,
            candidates,
            risks,
            collisions
        ) = plan(
            ego,
            agents
        )

        selected_paths.append(
            best_candidate["trajectory"].copy()
        )

        risk_history.append(
            risks.copy()
        )

        collision_history.append(
            collisions.copy()
        )

        # --------------------------------------------
        # APPLY ONLY CURRENT CONTROL
        # --------------------------------------------

        ego = ego_step(
            ego,
            best_candidate["target_y"],
            best_candidate["target_speed"]
        )

        # --------------------------------------------
        # UPDATE AGENTS
        # --------------------------------------------

        for agent in agents:

            agent.update(DT)

        ego_history.append([
            ego["x"],
            ego["y"]
        ])

        cow_history.append(
            agents[0].position.copy()
        )

    return (
        np.array(ego_history),
        np.array(cow_history),
        selected_paths,
        risk_history,
        collision_history
    )


# ============================================================
# CREATE RISK VISUALIZATION
# ============================================================

def calculate_risk_field(
    ego_position,
    cow_position
):

    x_values = np.linspace(
        -5,
        5,
        80
    )

    y_values = np.linspace(
        0,
        35,
        120
    )

    X, Y = np.meshgrid(
        x_values,
        y_values
    )

    dx = (
        X
        - cow_position[0]
    )

    dy = (
        Y
        - cow_position[1]
    )

    lateral_radius = (
        EGO_RADIUS
        + COW_RADIUS
    )

    forward_radius = (
        2.0
        + COW_RADIUS
    )

    distance_normalized = np.sqrt(

        (
            dx
            / lateral_radius
        ) ** 2

        +

        (
            dy
            / forward_radius
        ) ** 2
    )

    risk = np.exp(
        -0.5
        * (
            (
                distance_normalized
                - 1.0
            )
            / 0.35
        ) ** 2
    )

    return (
        X,
        Y,
        risk
    )


# ============================================================
# ANIMATION
# ============================================================

def create_animation(show=True):

    print(
        "Running baseline..."
    )

    (
        baseline_ego,
        baseline_cow
    ) = run_baseline()

    print(
        "Running PathRakshak..."
    )

    (
        path_ego,
        path_cow,
        selected_paths,
        risk_history,
        collision_history
    ) = run_pathrakshak()

    print(
        "Creating animation..."
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 12)
    )

    baseline_ax = axes[0]
    path_ax = axes[1]

    # ========================================================
    # BASELINE AXIS
    # ========================================================

    baseline_ax.set_xlim(
        -5,
        5
    )

    baseline_ax.set_ylim(
        0,
        ROAD_LENGTH
    )

    baseline_ax.set_xlabel(
        "Lateral position (m)"
    )

    baseline_ax.set_ylabel(
        "Forward position (m)"
    )

    baseline_ax.set_title(
        "BASELINE PLANNER"
    )

    baseline_ax.grid(
        alpha=0.3
    )

    # ========================================================
    # PATHRAKSHAK AXIS
    # ========================================================

    path_ax.set_xlim(
        -5,
        5
    )

    path_ax.set_ylim(
        0,
        ROAD_LENGTH
    )

    path_ax.set_xlabel(
        "Lateral position (m)"
    )

    path_ax.set_ylabel(
        "Forward position (m)"
    )

    path_ax.set_title(
        "PATHRAKSHAK AI"
    )

    path_ax.grid(
        alpha=0.3
    )

    # ========================================================
    # STATIC EGO TRAJECTORIES
    # ========================================================

    baseline_ax.plot(
        baseline_ego[:, 0],
        baseline_ego[:, 1],
        linestyle="--",
        label="Baseline trajectory"
    )

    path_ax.plot(
        path_ego[:, 0],
        path_ego[:, 1],
        linestyle="--",
        label="PathRakshak trajectory"
    )

    # ========================================================
    # CURRENT EGO
    # ========================================================

    baseline_car = baseline_ax.scatter(
        [],
        [],
        s=150,
        label="Ego vehicle"
    )

    path_car = path_ax.scatter(
        [],
        [],
        s=150,
        label="Ego vehicle"
    )

    # ========================================================
    # COW
    # ========================================================

    baseline_cow_marker = baseline_ax.scatter(
        [],
        [],
        s=250,
        marker="o",
        label="Cow"
    )

    path_cow_marker = path_ax.scatter(
        [],
        [],
        s=250,
        marker="o",
        label="Cow"
    )

    # ========================================================
    # SELECTED PATH
    # ========================================================

    selected_line, = path_ax.plot(
        [],
        [],
        linewidth=3,
        label="Selected path"
    )

    # ========================================================
    # CANDIDATE PATHS
    # ========================================================

    candidate_lines = []

    for _ in range(99):

        line, = path_ax.plot(
            [],
            [],
            linewidth=0.5,
            alpha=0.25
        )

        candidate_lines.append(
            line
        )

    # ========================================================
    # RISK TEXT
    # ========================================================

    risk_text = path_ax.text(
        0.02,
        0.96,
        "",
        transform=path_ax.transAxes,
        verticalalignment="top",
        fontsize=10
    )

    # ========================================================
    # STATUS TEXT
    # ========================================================

    status_text = path_ax.text(
        0.02,
        0.89,
        "",
        transform=path_ax.transAxes,
        verticalalignment="top",
        fontsize=10
    )

    baseline_ax.legend()
    path_ax.legend()

    # ========================================================
    # UPDATE
    # ========================================================

    def update(frame):

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        baseline_car.set_offsets([
            baseline_ego[frame]
        ])

        baseline_cow_marker.set_offsets([
            baseline_cow[frame]
        ])

        # ----------------------------------------------------
        # PATHRAKSHAK
        # ----------------------------------------------------

        path_car.set_offsets([
            path_ego[frame]
        ])

        path_cow_marker.set_offsets([
            path_cow[frame]
        ])

        # ----------------------------------------------------
        # SELECTED PATH
        # ----------------------------------------------------

        selected_path = (
            selected_paths[frame]
        )

        selected_line.set_data(
            selected_path[:, 0],
            selected_path[:, 1]
        )

        # ----------------------------------------------------
        # CANDIDATE PATHS
        # ----------------------------------------------------

        # Recalculate candidates from current
        # approximate state.

        current_state = {

            "x": path_ego[frame, 0],

            "y": path_ego[frame, 1],

            "vx": 0.0,

            "vy": 5.0,

            "ax": 0.0,

            "ay": 0.0
        }

        candidates = rollout_candidates(
            current_state
        )

        risks = risk_history[
            frame
        ]

        max_risk = np.max(
            risks
        )

        min_risk = np.min(
            risks
        )

        # Normalize risk for visibility

        if max_risk > 0:

            normalized_risk = (
                risks / max_risk
            )

        else:

            normalized_risk = (
                np.zeros_like(risks)
            )

        for i, candidate in enumerate(
            candidates
        ):

            trajectory = (
                candidate["trajectory"]
            )

            candidate_lines[i].set_data(
                trajectory[:, 0],
                trajectory[:, 1]
            )

            # Higher risk = more visually prominent

            candidate_lines[i].set_alpha(
                0.15
                +
                0.45
                * normalized_risk[i]
            )

        # ----------------------------------------------------
        # RISK FIELD
        # ----------------------------------------------------

        X, Y, risk_field = (
            calculate_risk_field(
                path_ego[frame],
                path_cow[frame]
            )
        )

        # Remove previous contour collections

        for collection in list(
            path_ax.collections
        ):

            if collection not in [
                path_car,
                path_cow_marker
            ]:

                try:

                    collection.remove()

                except:

                    pass

        path_ax.contourf(
            X,
            Y,
            risk_field,
            levels=12,
            alpha=0.12
        )

        # ----------------------------------------------------
        # RISK TEXT
        # ----------------------------------------------------

        risk_text.set_text(

            f"Minimum risk: "
            f"{min_risk:.3f}\n"

            f"Maximum risk: "
            f"{max_risk:.3f}"
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        collision_count = np.min(
            collision_history[frame]
        )

        if collision_count > 0:

            status = (
                "STATUS: COLLISION RISK"
            )

        else:

            status = (
                "STATUS: SAFE PATH"
            )

        status_text.set_text(
            status
        )

        return (
            baseline_car,
            baseline_cow_marker,
            path_car,
            path_cow_marker,
            selected_line,
            risk_text,
            status_text,
            *candidate_lines
        )

    # ========================================================
    # CREATE ANIMATION
    # ========================================================

    animation = FuncAnimation(

        fig,

        update,

        frames=SIMULATION_STEPS,

        interval=100,

        blit=False
    )

    # ========================================================
    # SAVE
    # ========================================================

    animation.save(

        "pathrakshak_stage8.gif",

        writer=PillowWriter(
            fps=10
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "STAGE 8 COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        "Saved:"
    )

    print(
        "pathrakshak_stage8.gif"
    )

    if show:
        try:
            plt.show(block=True)
        except Exception:
            plt.close(fig)
    else:
        plt.close(fig)


# ============================================================
# PERCEPTION ENTRY
# ============================================================

def find_video_path(explicit_path=None):

    candidate_paths = []

    if explicit_path:
        candidate_paths.append(explicit_path)

    candidate_paths.extend([
        "road_video.mp4",
        os.path.join("preception", "road_video.mp4"),
        os.path.join("perception", "road_video.mp4"),
        os.path.join("videos", "road_video.mp4"),
        os.path.join("assets", "road_video.mp4"),
    ])

    for path in candidate_paths:
        if path and os.path.exists(path):
            return path

    return None


def run_perception_demo(video_path=None, output_path=None):

    input_video = find_video_path(video_path)

    if input_video is None:
        print(
            "No video file found for perception demo. "
            "Add road_video.mp4 or pass --video <path>."
        )
        return None

    if output_path is None:
        output_path = os.path.join(
            "outputs",
            "pathrakshak_perception.mp4"
        )

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    try:
        from perception.detector import process_video
    except ImportError:
        from preception.detector import process_video

    process_video(
        input_video,
        output_path
    )

    print(
        f"Perception demo complete. Output: {output_path}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="PathRakshak AI entry point"
    )

    parser.add_argument(
        "--vision-demo",
        action="store_true",
        help="Run the perception detector on a video"
    )

    parser.add_argument(
        "--video",
        default=None,
        help="Optional input video path for the vision demo"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the vision demo"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the simulation benchmark"
    )

    args = parser.parse_args()

    if args.vision_demo:
        run_perception_demo(
            args.video,
            args.output
        )
        return

    if args.benchmark or not args.vision_demo:
        results = run_benchmark()
        print_results(
            results
        )
        plot_example(
            results
        )


if __name__ == "__main__":

    main()