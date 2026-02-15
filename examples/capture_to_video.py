import logging
import time

import rapidshot

# install OpenCV using `pip install rapidshot[cv2]` command.
import cv2

TOP = 0
LEFT = 0
RIGHT = 1920
BOTTOM = 1080
target_fps = 30
total_frames = 1500
output_path = "video.mp4"
output_size = (RIGHT - LEFT, BOTTOM - TOP)
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[Rapidshot] Capture benchmark"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting capture-to-video example")
    logger.info(
        "Config: region=%s output_size=%s target_fps=%d total_frames=%d output=%s",
        region,
        output_size,
        target_fps,
        total_frames,
        output_path,
    )

    # Use output_idx=1 which is 1920x1080 (see rapidshot.output_info())
    screencapture = rapidshot.create(
        device_idx=0, output_idx=1, output_color="RGB", nvidia_gpu=True
    )
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        output_size,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    start_time = time.time()
    logger.info("Capture started")

    try:
        screencapture.start(target_fps=target_fps, video_mode=True)
        for i in range(total_frames):
            writer.write(screencapture.get_latest_frame())
            if (i + 1) % 300 == 0 or (i + 1) == total_frames:
                logger.info("Progress: wrote %d/%d frames", i + 1, total_frames)
    finally:
        screencapture.stop()
        writer.release()
        elapsed = time.time() - start_time
        logger.info("Capture stopped and writer released")
        logger.info(
            "Done: wrote %d frames to %s in %.2fs (avg %.2f fps)",
            total_frames,
            output_path,
            elapsed,
            total_frames / elapsed if elapsed > 0 else 0.0,
        )


if __name__ == "__main__":
    main()
