import subprocess
import sys

def cleanup_ollama():
    try:
        # Get list of models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: Could not list Ollama models. Is Ollama installed?")
            return

        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            print("No models found to clean up.")
            return

        # Header is usually NAME ID SIZE MODIFIED
        # Skip the first line (header)
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            model_name = parts[0]
            
            # Keep llama3.2 (including latest or other tags)
            if 'llama3.2' in model_name:
                print(f"Keeping required model: {model_name}")
            else:
                print(f"Removing unwanted model: {model_name}...")
                subprocess.run(['ollama', 'rm', model_name])

        print("Cleanup complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    cleanup_ollama()
