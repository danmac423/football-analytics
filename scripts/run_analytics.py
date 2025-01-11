import argparse

from football_analytics.annotations.annotations import (
    run_pitch_keypoints_detection,
    run_player_detection,
)
from football_analytics.utils.utils import create_video_sink, video_frames_generator


def run_analytics(source_video_path, output_video_path):
    out = create_video_sink(source_video_path, output_video_path)

    for frame in video_frames_generator(source_video_path):
        frame_with_players = next(run_player_detection(frame))

        frame_with_annotations = next(run_pitch_keypoints_detection(frame_with_players))

        out.write(frame_with_annotations)

    out.release()
    print(f"Video saved to {output_video_path}")



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

    run_analytics(source_video_path, output_video_path)

