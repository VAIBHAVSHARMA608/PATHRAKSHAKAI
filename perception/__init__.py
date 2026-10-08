"""Perception package for PathRakshak AI."""

from .detector import (
    detect_frame,
    detections_to_agents,
    process_image_stream,
    process_video,
)

__all__ = [
    "detect_frame",
    "detections_to_agents",
    "process_image_stream",
    "process_video",
]
