import cv2
import base64
import os
import json
import requests
from typing import List, Dict, Optional
import numpy as np


def extract_frames(
    video_path: str, num_frames: int = 10, output_dir: Optional[str] = None
) -> List[np.ndarray]:
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        output_dir: Optional directory to save extracted frames as images

    Returns:
        List of numpy arrays containing the extracted frames
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Video duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")

    # Calculate frame indices to extract
    if total_frames <= num_frames:
        # If video has fewer frames than requested, take all frames
        frame_indices = list(range(total_frames))
    else:
        # Otherwise, take evenly spaced frames
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    # Extract the frames
    frames = []
    for idx in frame_indices:
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Read the frame
        success, frame = video.read()

        if success:
            frames.append(frame)

            # Save frame if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                frame_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"Saved frame to {frame_path}")
        else:
            print(f"Failed to read frame at index {idx}")

    # Release the video capture object
    video.release()

    return frames


def frames_to_base64(frames: List[np.ndarray], format: str = "jpg") -> List[str]:
    """
    Convert a list of frames to base64-encoded strings.

    Args:
        frames: List of numpy arrays containing the frames
        format: Image format to encode the frames (jpg, png, etc.)

    Returns:
        List of base64-encoded strings
    """
    encoded_frames = []

    for frame in frames:
        # Encode frame as an image
        success, buffer = cv2.imencode(f".{format}", frame)

        if success:
            # Convert to base64
            encoded_frame = base64.b64encode(buffer).decode("utf-8")
            encoded_frames.append(encoded_frame)
        else:
            print("Failed to encode frame")

    return encoded_frames


if __name__ == "__main__":
    frames = extract_frames(
        "/home/edward/Documents/Code/ScAI/AccountingProj/generated_visuals/pythagorean_theorem_animation.mp4",
        5,
    )
    print(f"Extracted {len(frames)} frames")

    # Convert to base64
    encoded_frames = frames_to_base64(frames)
