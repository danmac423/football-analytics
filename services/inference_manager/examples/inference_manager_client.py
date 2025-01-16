import cv2
import grpc
import numpy as np

from config import INFERENCE_MANAGER_SERVICE_ADDRESS
from services.inference_manager.grpc_files import inference_manager_pb2, inference_manager_pb2_grpc


def frame_generator(video_path: str):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        yield inference_manager_pb2.Frame(frame_id=frame_id, content=buffer.tobytes(), fps=fps)

        frame_id += 1

    cap.release()


def run_client(video_path: str):
    with grpc.insecure_channel(INFERENCE_MANAGER_SERVICE_ADDRESS) as channel:
        stub = inference_manager_pb2_grpc.InferenceManagerServiceStub(channel)

        responses = stub.ProcessFrames(frame_generator(video_path))

        for response in responses:
            print(f"Received annotated frame ID: {response.frame_id}")

            annotated_frame = cv2.imdecode(
                np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
            )

            cv2.imshow("Annotated Frame", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "data/test.mp4"
    run_client(video_path)
