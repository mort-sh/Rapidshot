"""
Wrapper script to run RapidShot benchmarks.
This avoids issues with bash stdin redirection on Windows.
"""

import subprocess
import sys

if __name__ == "__main__":
    # Run the benchmark directly
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    benchmark_script = "benchmarks/rapidshot_max_fps.py"

    cmd = [sys.executable, benchmark_script] + args
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)
