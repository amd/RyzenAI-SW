import subprocess
import os

def run_batch_file(bat_file, timeout=300):
    """
    Runs a batch file with a timeout.

    Args:
    bat_file (str): Path to the batch file.
    timeout (int): Time in seconds before forcefully terminating the process.

    Returns:
    None
    """
    try:
        # Get the directory of the batch file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bat_path = os.path.join(script_dir, bat_file)

        if not os.path.exists(bat_path):
            print(f"ERROR: The batch file '{bat_file}' does not exist in {script_dir}")
            return

        print(f"Running: {bat_path} (Timeout: {timeout} seconds)")
        
        # Run the batch file with timeout
        process = subprocess.run(bat_path, shell=True, timeout=timeout, check=True)

        print("Batch file executed successfully!")

    except subprocess.TimeoutExpired:
        print(f"ERROR: Batch file execution timed out after {timeout} seconds.")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Batch file execution failed with error: {e}")

    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    run_batch_file("run_modelx4.bat", timeout=300)  # Adjust timeout as needed
