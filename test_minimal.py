import rapidshot

print("Test 1: Import and version info")
print(rapidshot.get_version_info())

print("\nTest 2: Create screencapture")
try:
    screencapture = rapidshot.create(output_color='RGB', nvidia_gpu=False)
    print("Success: Screencapture created")

    print("\nTest 3: Grab one frame")
    frame = screencapture.grab((0, 0, 100, 100))
    if frame is not None:
        print(f"Success: Got frame with shape {frame.shape}")
    else:
        print("Warning: Frame is None")

    print("\nTest 4: Release")
    screencapture.release()
    print("Success: Released screencapture")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest 5: Clean up")
rapidshot.clean_up()
print("Done!")
