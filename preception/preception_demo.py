from perception.detector import (
    process_video
)


VIDEO_PATH = "road_video.mp4"

OUTPUT_PATH = (
    "pathrakshak_perception.mp4"
)


process_video(
    VIDEO_PATH,
    OUTPUT_PATH
)

print(
    "Perception demo completed."
)

print(
    f"Output: {OUTPUT_PATH}"
)