"""Runnable continuous-capture example for RapidShot.

Usage:
    uv run python continuious_example.py
    uv run python continuious_example.py --frames 500 --fps 120
"""

from __future__ import annotations

import argparse
import time

import rapidshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RapidShot continuous-capture smoke test.")
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target capture FPS passed to screencapture.start().",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Number of frames to read before stopping.",
    )
    parser.add_argument(
        "--device-idx",
        type=int,
        default=0,
        help="GPU adapter index.",
    )
    parser.add_argument(
        "--output-idx",
        type=int,
        default=None,
        help="Display output index. Default auto-selects primary output.",
    )
    parser.add_argument(
        "--output-color",
        type=str,
        default="BGRA",
        choices=["RGB", "RGBA", "BGR", "BGRA", "GRAY"],
        help="Frame color format.",
    )
    parser.add_argument(
        "--video-mode",
        action="store_true",
        help="Enable video mode (duplicate last frame if no new frame arrives).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    screencapture = None
    collected = 0
    missing = 0
    first_frame_shape = None

    print("RapidShot device info:")
    print(rapidshot.device_info().strip() or "(no devices listed)")
    print("RapidShot output info:")
    print(rapidshot.output_info().strip() or "(no outputs listed)")

    start_time = time.perf_counter()
    try:
        screencapture = rapidshot.create(
            device_idx=args.device_idx,
            output_idx=args.output_idx,
            output_color=args.output_color,
        )
        screencapture.start(target_fps=args.fps, video_mode=args.video_mode)

        while collected < args.frames:
            frame = screencapture.get_latest_frame()
            if frame is None:
                missing += 1
                continue

            if first_frame_shape is None:
                first_frame_shape = frame.shape
                print(f"First frame shape: {first_frame_shape}")

            collected += 1

        elapsed = time.perf_counter() - start_time
        achieved_fps = collected / elapsed if elapsed > 0 else 0.0
        print(f"Captured {collected} frames in {elapsed:.2f}s")
        print(f"Approx achieved read FPS: {achieved_fps:.2f}")
        print(f"Frame wait timeouts (None returns): {missing}")
        return 0
    finally:
        if screencapture is not None:
            try:
                screencapture.stop()
            except Exception:
                pass
            try:
                screencapture.release()
            except Exception:
                pass
        rapidshot.clean_up()


if __name__ == "__main__":
    raise SystemExit(main())
