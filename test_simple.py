import time
import rapidshot

print("Creating screencapture...")
screencapture = rapidshot.create(output_color="RGB", nvidia_gpu=False)

region = (0, 0, 1920, 1080)
print(f"Region: {region}")

print("Testing 10 grabs...")
for i in range(10):
    frame = screencapture.grab(region=region)
    if frame is not None:
        print(f"Frame {i + 1}: shape={frame.shape}")
    else:
        print(f"Frame {i + 1}: None")

print("Done!")
screencapture.release()
rapidshot.clean_up()
