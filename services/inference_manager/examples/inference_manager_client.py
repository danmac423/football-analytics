from pathlib import Path

import cv2
import grpc
import click
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


@click.command()
@click.option(
    "--interactive-mode",
    is_flag=True,
    help="Run the client in interactive mode to process a video file."
)
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
def main(interactive_mode, video_path):
    """
    CLI application to process video files with the gRPC inference service.

    VIDEO_PATH: Path to the mp4 video file to be processed.
    """
    if interactive_mode:
        click.echo("Interactive mode enabled.")
        click.echo(f"Processing video: {video_path}")
        run_client(video_path)
    else:
        click.echo("Interactive mode not enabled. No action taken.")



if __name__ == "__main__":
    main()
