from pathlib import Path

import cv2
import grpc
import click
import numpy as np
from multimethod import multimethod

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


@multimethod
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


@multimethod
def run_client(video_path: str, output_path: str):
    # Open video capture to get video properties
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Initialize VideoWriter to save the annotated frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with grpc.insecure_channel(INFERENCE_MANAGER_SERVICE_ADDRESS) as channel:
        stub = inference_manager_pb2_grpc.InferenceManagerServiceStub(channel)

        responses = stub.ProcessFrames(frame_generator(video_path))

        for response in responses:
            print(f"Received annotated frame ID: {response.frame_id}")

            # Decode the annotated frame
            annotated_frame = cv2.imdecode(
                np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
            )

            # Save the annotated frame to the output video
            out.write(annotated_frame)

    # Release the VideoWriter
    out.release()
    print(f"Annotated video saved to {output_path}")


@click.command()
@click.option(
    "--interactive-mode",
    is_flag=True,
    help="Run the client in interactive mode to process a video file."
)
@click.option(
    "--save-to-file-mode",
    type=click.Path(writable=True, dir_okay=False),
    default=None,
    help="Run the client in save-to-file mode and save the annotated video to the specified path."
)
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
def main(interactive_mode, save_to_file_mode, video_path):
    """
    CLI application to process video files with the gRPC inference service.

    VIDEO_PATH: Path to the mp4 video file to be processed.
    """
    if interactive_mode and save_to_file_mode:
        raise click.UsageError("You cannot use --interactive-mode and --save-to-file-mode together.")

    if interactive_mode:
        click.echo("Interactive mode enabled.")
        click.echo(f"Processing video: {video_path}")
        run_client(video_path)
    elif save_to_file_mode:
        click.echo("Save-to-file mode enabled.")
        click.echo(f"Processing video: {video_path}")
        click.echo(f"Saving output video to: {save_to_file_mode}")
        run_client(video_path, save_to_file_mode)
    else:
        click.echo("No mode selected. Use --interactive-mode or --save-to-file-mode.")


if __name__ == "__main__":
    main()
