import os
import signal
import subprocess
import time

def kill_python_processes():
    print("Identifying python processes...")
    try:
        # Get current process ID
        my_pid = os.getpid()
        
        # Use tasklist to find other python processes
        output = subprocess.check_output('tasklist /FI "IMAGENAME eq python.exe" /FO CSV', shell=True).decode()
        lines = output.strip().split('\n')[1:] # Skip header
        
        for line in lines:
            parts = line.split(',')
            if len(parts) > 1:
                pid = int(parts[1].strip('"'))
                if pid != my_pid:
                    print(f"Killing process {pid}...")
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except:
                            pass
    except Exception as e:
        print(f"Error identification: {e}")

def force_delete_files(directory):
    print(f"Cleaning {directory}...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "upsample_40m.pt" in file:
                path = os.path.join(root, file)
                print(f"Deleting {path}...")
                for attempt in range(5):
                    try:
                        os.remove(path)
                        print("Deleted.")
                        break
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed: {e}")
                        time.sleep(1)

if __name__ == "__main__":
    kill_python_processes()
    time.sleep(2)
    force_delete_files("backend/point_e_model_cache")
    print("Cleanup complete.")
