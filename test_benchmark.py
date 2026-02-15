import time
import rapidshot

# Setup capture region
region = (0, 0, 100, 100)
print(f"Starting benchmark test")
print(f"Region: {region}")

# Benchmark code
start_time = time.perf_counter()
fps = 0

# Create screencapture
print("Creating screencapture...")
screencapture = rapidshot.create(output_color='RGB', nvidia_gpu=False)
print("Screencapture created")

# Warm-up
print("Warming up...")
_ = screencapture.grab(region=region)
time.sleep(0.1)

# Run benchmark
print("Running benchmark (50 frames)...")
max_frames = 50
while fps < max_frames:
    frame = screencapture.grab(region=region)
    if frame is not None:
        fps += 1
        if fps % 10 == 0:
            print(f"  Captured {fps} frames...")

end_time = time.perf_counter() - start_time
fps_rate = fps / end_time

print(f"\nResults:")
print(f"- Total frames: {fps}")
print(f"- Time elapsed: {end_time:.2f} seconds")
print(f"- Average FPS: {fps_rate:.2f}")

# Clean up
print("Cleaning up...")
del screencapture
rapidshot.clean_up()
print("Done!")
