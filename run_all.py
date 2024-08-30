import subprocess
import sys
import os

def run_script(script_path):
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        sys.exit(1)

def main():
    # Base directory (assuming this script is in the base directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # List your scripts in the order you want them to run
    scripts = [
        os.path.join(base_dir, "Baseline(0-shot and K-shot).py"),
        os.path.join(base_dir, "Rubin Approach", "Rubin_data_creation.py"),
        os.path.join(base_dir, "Rubin Approach", "Rubin_training.py"),
        os.path.join(base_dir, "Rubin Approach", "Rubin_inference.py"),
        os.path.join(base_dir, "Our Approach", "Our_approach_single_shot.py"),
        os.path.join(base_dir, "Our Approach", "Our_approach_data_creation.py"),
        os.path.join(base_dir, "Our Approach", "Our_approach_training.py"),
        os.path.join(base_dir, "Our Approach", "Our_approach_inference.py")
    ]

    for script in scripts:
        print(f"Running {script}...")
        run_script(script)
        print(f"{script} completed.\n")

    print("All scripts have been executed successfully.")

if __name__ == "__main__":
    main()
