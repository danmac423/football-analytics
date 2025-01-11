import cv2
import numpy as np
import supervision as sv
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image
from ultralytics import YOLO  # type: ignore

from football_analytics.utils.utils import serialize_object

infer_router = APIRouter()

model_players_detection = YOLO("models/football-player-detector-n.pt").to("mps")
model_pitch_keypoints_detection = YOLO("models/football-pitch-keypoints-detector-x.pt").to("mps")

@infer_router.post("/infer")
async def infer(
    file: UploadFile = File(...),
    height: int = Form(...),
    width: int = Form(...),
    channels: int = Form(...),
):
    try:
        byte_stream = await file.read()
        np_array = np.frombuffer(byte_stream, np.uint8)
        np_array = np_array.reshape((height, width, channels))

        image = Image.fromarray(cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB))

        result = model_players_detection.predict(image, save=False)[0]

        detections = sv.Detections.from_ultralytics(result)

        response = serialize_object(detections)
        return response

    except Exception as e:
        print(e)


@infer_router.post("/keypoints")
async def keypoints_infer(
    file: UploadFile = File(...),
    height: int = Form(...),
    width: int = Form(...),
    channels: int = Form(...),
):
    try:
        byte_stream = await file.read()
        np_array = np.frombuffer(byte_stream, np.uint8)
        np_array = np_array.reshape((height, width, channels))

        image = Image.fromarray(cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB))

        result = model_pitch_keypoints_detection.predict(image, mode="pose", imgsz=960)[0]

        detections = sv.Detections.from_ultralytics(result)
        keypoints = sv.KeyPoints.from_ultralytics(result)

        response = {
            "detections": serialize_object(detections),
            "keypoints": serialize_object(keypoints),
        }
        return response

    except Exception as e:
        print(e)
        return {"error": str(e)}