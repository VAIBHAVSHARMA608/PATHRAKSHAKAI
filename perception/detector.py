import os

import cv2
from ultralytics import YOLO

from dataset.realtime import iter_image_frames
from simulator.agent import Agent


MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)


TRACKABLE_CLASSES = {
    "person": 0.45,
    "bicycle": 0.7,
    "motorcycle": 0.8,
    "car": 1.0,
    "bus": 1.4,
    "truck": 1.5,
}


def detect_frame(frame, min_confidence=0.35):
    """Return planner-neutral detections for one camera frame."""

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )[0]
    boxes = result.boxes
    names = result.names
    detections = []

    if boxes is None:
        return detections

    coordinates = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    track_ids = (
        boxes.id.int().cpu().tolist()
        if boxes.id is not None
        else [None] * len(coordinates)
    )

    for box, confidence, class_id, track_id in zip(
        coordinates, confidences, classes, track_ids
    ):
        label = names[int(class_id)]
        if confidence < min_confidence or label not in TRACKABLE_CLASSES:
            continue
        detections.append(
            {
                "label": label,
                "confidence": float(confidence),
                "bbox": [float(value) for value in box],
                "track_id": track_id,
            }
        )

    return detections


def detections_to_agents(detections, frame_width, frame_height):
    """Map 2D detections to approximate planner agents for replay demos.

    This uses a normalized pinhole-road approximation. A calibrated camera or
    depth sensor should replace it for real vehicle deployment.
    """

    agents = []
    for detection in detections:
        left, top, right, bottom = detection["bbox"]
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        lateral_position = (center_x / max(frame_width, 1) - 0.5) * 8.0
        forward_position = 2.0 + (1.0 - center_y / max(frame_height, 1)) * 28.0
        agents.append(
            Agent(
                position=[lateral_position, forward_position],
                velocity=[0.0, 0.0],
                radius=TRACKABLE_CLASSES[detection["label"]],
            )
        )
    return agents


def process_video(video_path, output_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        success, frame = video.read()
        if not success:
            break

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        annotated_frame = result.plot()
        writer.write(annotated_frame)

        cv2.imshow("PathRakshak Perception", annotated_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()


def process_image_stream(
    image_dir="dataset/idd20k_lite",
    split="val",
    limit=20,
    output_path=None,
    display=False,
):
    """Run the same tracker over the checked-in IDD images in realtime order.

    The image dataset is treated as a camera stream. When ``output_path`` is
    omitted, frames are processed without writing a video file.
    """

    image_paths = iter_image_frames(image_dir, split)
    writer = None
    processed = 0

    try:
        for image_path in image_paths:
            if limit is not None and processed >= limit:
                break

            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
            annotated_frame = result.plot()

            if output_path and writer is None:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                height, width = annotated_frame.shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (width, height),
                )

            if writer is not None:
                writer.write(annotated_frame)

            if display:
                cv2.imshow("PathRakshak Perception", annotated_frame)
                if cv2.waitKey(1) == 27:
                    break

            processed += 1
    finally:
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    return processed
