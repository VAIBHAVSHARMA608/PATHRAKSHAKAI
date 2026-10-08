import cv2

from ultralytics import YOLO


# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_NAME = "yolo11n.pt"

model = YOLO(
    MODEL_NAME
)


# --------------------------------------------------
# DETECTION + TRACKING
# --------------------------------------------------

def process_video(
    video_path,
    output_path
):

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        raise RuntimeError(
            "Could not open video"
        )

    width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30

    writer = cv2.VideoWriter(

        output_path,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width, height)
    )

    while True:

        success, frame = (
            video.read()
        )

        if not success:

            break

        # ------------------------------------------
        # YOLO + BYTE TRACK
        # ------------------------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            verbose=False
        )

        result = results[0]

        # ------------------------------------------
        # DRAW RESULTS
        # ------------------------------------------

        annotated_frame = (
            result.plot()
        )

        writer.write(
            annotated_frame
        )

        cv2.imshow(
            "PathRakshak Perception",
            annotated_frame
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    video.release()

    writer.release()

    cv2.destroyAllWindows()