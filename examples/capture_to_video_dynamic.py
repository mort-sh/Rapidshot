import re

import cv2
import rapidshot


# install OpenCV using `pip install rapidshot[cv2]` command.
OUTPUT_LINE_PATTERN = re.compile(
    r"^Device\[(?P<device_idx>\d+)\]\s+Output\[(?P<output_idx>\d+)\]:"
    r"\s+Resolution:\((?P<width>\d+),\s*(?P<height>\d+)\)"
)


def parse_outputs(raw_output_info: str):
    """Parse rapidshot.output_info() lines into structured output metadata."""
    parsed_outputs = []
    for line in raw_output_info.splitlines():
        text = line.strip()
        if not text:
            continue
        match = OUTPUT_LINE_PATTERN.match(text)
        if not match:
            continue

        parsed_outputs.append(
            {
                "device_idx": int(match.group("device_idx")),
                "output_idx": int(match.group("output_idx")),
                "width": int(match.group("width")),
                "height": int(match.group("height")),
                "line": text,
            }
        )
    return parsed_outputs


def format_available_outputs(outputs):
    if not outputs:
        return "  (none)"
    return "\n".join(f"  - {entry['line']}" for entry in outputs)


def main():
    # Get output information to determine resolution.
    output_idx = 0  # Change this to select different monitor.
    target_fps = 30
    total_frames = 600

    raw_outputs = rapidshot.output_info()
    parsed_outputs = parse_outputs(raw_outputs)
    if not parsed_outputs:
        raise RuntimeError(
            "No outputs were detected from rapidshot.output_info(). "
            "Check your display configuration and GPU compatibility."
        )

    target_output = next(
        (entry for entry in parsed_outputs if entry["output_idx"] == output_idx), None
    )
    if target_output is None:
        raise ValueError(
            f"Output index {output_idx} was not found.\n"
            f"Available outputs:\n{format_available_outputs(parsed_outputs)}\n"
            "Update output_idx or inspect rapidshot.output_info() for valid values."
        )

    width = target_output["width"]
    height = target_output["height"]
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid output resolution detected for output {output_idx}: {width}x{height}"
        )

    print(f"Capturing from output {output_idx} at {width}x{height}")

    screencapture = None
    writer = None
    try:
        screencapture = rapidshot.create(output_idx=output_idx, output_color="BGR")
        screencapture.start(target_fps=target_fps, video_mode=True)

        writer = cv2.VideoWriter(
            "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to initialize OpenCV VideoWriter for video.mp4.")

        print(f"Recording {total_frames / target_fps:.1f} seconds of video...")
        for i in range(total_frames):
            frame = screencapture.get_latest_frame()
            if frame is not None:
                writer.write(frame)
            if (i + 1) % target_fps == 0:
                print(f"  Frame {i + 1}/{total_frames}")

        print("Video saved to video.mp4")
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as exc:
        print(f"Recording failed: {exc}")
        raise
    finally:
        if screencapture is not None:
            try:
                screencapture.stop()
            except Exception as stop_error:
                print(f"Warning: failed to stop capture cleanly: {stop_error}")

        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
