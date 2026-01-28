import subprocess

try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
except Exception as e:
    print(f"Error: {e}")
