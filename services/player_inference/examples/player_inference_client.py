import cv2
import grpc

from config import PLAYER_INFERENCE_SERVICE_ADDRESS
from services.player_inference.grpc_files import player_inference_pb2, player_inference_pb2_grpc


def stream_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        yield player_inference_pb2.Frame(frame_id=frame_id, content=buffer.tobytes())
        frame_id += 1
    cap.release()


def run_client(video_path, output_path):
    channel = grpc.insecure_channel(PLAYER_INFERENCE_SERVICE_ADDRESS)
    stub = player_inference_pb2_grpc.YOLOPlayerInferenceServiceStub(channel)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    responses = stub.InferencePlayers(stream_frames(video_path))
    for response in responses:
        ret, frame = cap.read()
        if not ret:
            break

        for box in response.boxes:
            x1 = int(box.x1_n * width)
            y1 = int(box.y1_n * height)
            x2 = int(box.x2_n * width)
            y2 = int(box.y2_n * height)
            confidence = box.confidence
            label = box.class_label
            tracker_id = box.tracker_id

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f}) {tracker_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        out.write(frame)

        cv2.imshow("Annotated Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Naciśnij 'q', aby przerwać
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # print(f"Frame ID: {response.frame_id}")
    # for box in response.boxes:
    #     print(f"  Bounding Box: ({box.x1_n}, {box.y1_n}, {box.x2_n}, {box.y2_n})")
    #     print(f"  Confidence: {box.confidence}, Class: {box.class_label}")


if __name__ == "__main__":
    run_client("data/test.mp4", "data/output/player_inference.mp4")
