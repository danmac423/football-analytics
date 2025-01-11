import argparse

from football_analytics.annotations.annotations import (
    run_pitch_keypoints_detection,
    run_player_detection,
)
from football_analytics.utils.utils import create_video_sink


def run_player_annotations(source_video_path, output_video_path):
    out = create_video_sink(source_video_path, output_video_path)

    for frame in run_player_detection(source_video_path):
        out.write(frame)

    out.release()

def run_pitch_keypoints_annotations(source_video_path, output_video_path):
    out = create_video_sink(source_video_path, output_video_path)

    for frame in run_pitch_keypoints_detection(source_video_path):
        out.write(frame)

    out.release()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_video_path",
        type=str,
        help="Path to the input video file",
    )
    parser.add_argument(
        "output_video_path",
        type=str,
        help="Path to the output video file",
    )
    args = parser.parse_args()
    source_video_path = args.source_video_path
    output_video_path = args.output_video_path

    # run_player_annotations(source_video_path, output_video_path)
    run_pitch_keypoints_annotations(source_video_path, output_video_path)

    print(f"Video saved to {output_video_path}")
