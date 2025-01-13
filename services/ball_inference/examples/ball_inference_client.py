import cv2
import grpc

from services.ball_inference.grpc_files import ball_inference_pb2, ball_inference_pb2_grpc


def stream_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        yield ball_inference_pb2.Frame(frame_id=frame_id, content=buffer.tobytes())
        frame_id += 1
    cap.release()


def run_client(video_path, output_path):
    channel = grpc.insecure_channel("localhost:50051")
    stub = ball_inference_pb2_grpc.YOLOBallInferenceServiceStub(channel)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    responses = stub.InferenceBall(stream_frames(video_path))
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        out.write(frame)

        cv2.imshow("Annotated Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_client("data/test.mp4", "data/ball_inference.mp4")
