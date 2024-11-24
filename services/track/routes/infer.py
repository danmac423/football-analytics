from fastapi import APIRouter, File, UploadFile, Form
from ultralytics import YOLO
from core.utils import serialize_object
import supervision as sv
from PIL import Image

import numpy as np
import cv2


infer_router = APIRouter()


model = YOLO("models/football-player-detector-s.pt").to("mps")


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

        result = model.predict(image, save=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        response = serialize_object(detections)
        print(response)

        return response

    except Exception as e:
        print(e)
