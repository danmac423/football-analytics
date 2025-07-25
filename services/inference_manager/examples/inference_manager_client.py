import click
import cv2
import grpc
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
def run_client(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with grpc.insecure_channel(INFERENCE_MANAGER_SERVICE_ADDRESS) as channel:
        stub = inference_manager_pb2_grpc.InferenceManagerServiceStub(channel)

        responses = stub.ProcessFrames(frame_generator(video_path))

        for response in responses:
            print(f"Received annotated frame ID: {response.frame_id}")

            annotated_frame = cv2.imdecode(
                np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
            )

            out.write(annotated_frame)

    out.release()
    print(f"Annotated video saved to {output_path}")


@multimethod  # type: ignore
def run_client(video_path: str):  # noqa
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
    help="Run the client in interactive mode to process a video file.",
)
@click.option(
    "--save-to-file-mode",
    is_flag=True,
    help="Run the client in save-to-file mode and save the annotated video to the specified path.",
)
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", required=False, type=click.Path(writable=True, dir_okay=False))
def main(interactive_mode, save_to_file_mode, video_path, output_path):
    """
    CLI application to process video files with the gRPC inference service.
    """
    if interactive_mode and save_to_file_mode:
        raise click.UsageError(
            "You cannot use --interactive-mode and --save-to-file-mode together."
        )

    if interactive_mode:
        if output_path:
            raise click.UsageError("Output path is not used in interactive mode.")
        click.echo("Interactive mode enabled.")
        click.echo(f"Processing video: {video_path}")
        run_client(video_path)
    elif save_to_file_mode:
        if not output_path:
            raise click.UsageError("Output path is required for --save-to-file-mode.")
        click.echo("Save-to-file mode enabled.")
        click.echo(f"Processing video: {video_path}")
        click.echo(f"Saving output video to: {output_path}")
        run_client(video_path, output_path)
    else:
        click.echo("No mode selected. Use --interactive-mode or --save-to-file-mode.")


if __name__ == "__main__":
    main()
