"""Generates a synthetic test video with simple face geometry for offline testing."""

import os
from pathlib import Path
import cv2
import numpy as np


def create_synthetic_video(output_path: str = "data/test_videos/sample.mp4", num_frames: int = 150) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

    for i in range(num_frames):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        # Draw a neutral stylized face
        cv2.ellipse(frame, (320, 240), (120, 160), 0, 0, 360, (200, 180, 160), -1)
        # Eyes
        eye_y = 210
        cv2.circle(frame, (270, eye_y), 15, (255, 255, 255), -1)
        cv2.circle(frame, (370, eye_y), 15, (255, 255, 255), -1)
        cv2.circle(frame, (270, eye_y), 6, (0, 0, 0), -1)
        cv2.circle(frame, (370, eye_y), 6, (0, 0, 0), -1)
        # Mouth
        cv2.line(frame, (290, 310), (350, 310), (50, 50, 180), 4)
        # Text
        cv2.putText(frame, f"Test Frame {i:03d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(frame)

    out.release()
    print(f"Created synthetic video at {output_path}")


if __name__ == "__main__":
    create_synthetic_video()
