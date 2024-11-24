from utils import video_frames_generator
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO


PLAYERS_DETECTION_MODEL = YOLO("models/football-player-detector-x.pt").to("mps")


def run_player_detection(source_video_path, player_label="player", score_threshold=0.1):
    """
    Annotates players in video frames
    """
    for frame in video_frames_generator(source_video_path):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = PLAYERS_DETECTION_MODEL(image, nms=False, conf=0.8, imgsz=1280)[0]
        detections = make_detections(results)

        image = draw_annotation(image, detections)

        annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        yield annotated_frame


def make_detections(results):
    """
    Extracts detections from the model results
    """

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            label = results.names[cls]
            detections.append((label, bbox, score))

    return detections


def draw_annotation(image, detections):
    """
    Draws annotations on the image
    """

    color = {"player": "red", "ball": "blue", "goalkeeper": "green", "referee": "yellow"}

    draw = ImageDraw.Draw(image)
    for label, bbox, score in detections:
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        y_bottom = bbox[3]
        y_top = bbox[1]

        radius = int(((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 4)
        # draw.ellipse(
        #     [x_center - radius, y_center - radius, x_center + radius, y_center + radius],
        #     outline=color[label],
        #     width=5,
        # )

        # draw.pieslice(
        #     [x_center - radius, y_bottom - radius, x_center + radius, y_bottom + radius],
        #     start=0,
        #     end=180,
        #     fill=None,
        #     outline=color[label],
        #     width=3,
        # )

        flag_width = int((bbox[2] - bbox[0]) * 0.6)
        flag_height = 20

        draw.polygon(
            [
                (x_center - flag_width / 2, y_top - 10 - flag_height),  # Lewy górny róg
                (x_center + flag_width / 2, y_top - 10 - flag_height),  # Prawy górny róg
                (x_center, y_top - 10),  # Wierzchołek flagi
            ],
            fill=color[label],
        )

        text_position = (x_center - flag_width / 4, y_bottom + 15)
        draw.text(text_position, f"{label}: {score:.2f}", fill="black")

    return image


if __name__ == "__main__":
    source_video_path = (
        "/Users/nataliapieczko/Desktop/studia/semestr5/ZPRP/football-analytics/test_video.mp4"
    )
    for annotated_frame in run_player_detection(source_video_path):
        cv2.imshow("Annotated Frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
