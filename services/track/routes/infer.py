from fastapi import APIRouter, File, UploadFile
from ultralytics import YOLO
from core.utils import serialize_object
import supervision as sv

import numpy as np
import cv2


infer_router = APIRouter()


model = YOLO("models/track/football-player-detector-n.pt").to("mps")


@infer_router.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        np_array = np.frombuffer(await file.read(), np.uint8)

        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        result = model.predict(image, save=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        response = serialize_object(detections)

        return response

    except Exception as e:
        print(e)
