from utils import video_frames_generator
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


PLAYERS_DETECTION_MODEL = YOLO("models/football-player-detector-s.pt").to("mps")

CUSTOM_PALLETE = sv.ColorPalette(
    {
        0: sv.Color(255, 0, 0),  # ball
        1: sv.Color(0, 0, 255),  # goalkeeper
        2: sv.Color(0, 255, 0),  # player
        3: sv.Color(255, 255, 0),  # referee
    }
)

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=CUSTOM_PALLETE)


def run_player_detection(source_video_path, score_threshold=0.1):
    """
    Annotates players in video frames
    """

    for frame in video_frames_generator(source_video_path):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = PLAYERS_DETECTION_MODEL(image, nms=False, conf=score_threshold, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)

        image = ELLIPSE_ANNOTATOR.annotate(image, detections)

        annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        yield annotated_frame


if __name__ == "__main__":
    source_video_path = (
        "/Users/nataliapieczko/Desktop/studia/semestr5/ZPRP/football-analytics/test_video.mp4"
    )
    for annotated_frame in run_player_detection(source_video_path):
        cv2.imshow("Annotated Frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
