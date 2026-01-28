import os
try:
    path = r"c:\hackathon\Gemini_CLI\Generative-Design\backend\test_dir"
    os.makedirs(path, exist_ok=True)
    print(f"Successfully created {path}")
except Exception as e:
    print(f"Failed: {e}")
