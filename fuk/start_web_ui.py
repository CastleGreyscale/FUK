#!/usr/bin/env python3
"""
FUK Web UI Launcher - Starts the web interface from the organized project structure
"""

import subprocess
import sys
from pathlib import Path
import time

def main():
    project_root = Path(__file__).parent
    ui_dir = project_root / "ui"
    server_script = ui_dir / "fuk_web_server.py"
    
    print("="*60)
    print("FUK Web UI Launcher")
    print("="*60)
    print()
    
    # Verify project structure
    required = {
        "core/": project_root / "core",
        "utils/": project_root / "utils",
        "ui/": ui_dir,
        "config/": project_root / "config",
        "vendor/musubi-tuner/": project_root / "vendor" / "musubi-tuner"
    }
    
    missing = []
    for name, path in required.items():
        if path.exists():
            print(f"✓ {name}")
        else:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print()
        print("ERROR: Missing required directories:")
        for name in missing:
            print(f"  - {name}")
        print()
        print("Please ensure you have the complete FUK project structure.")
        return 1
    
    print()
    print("="*60)
    print("Starting backend server...")
    print("="*60)
    print()
    print(f"Server script: {server_script}")
    print(f"Working directory: {project_root}")
    print()
    print("The server will start on http://localhost:8000")
    print("Open http://localhost:3000 in your browser to access the UI")
    print()
    print("Press Ctrl+C to stop the server")
    print("="*60)
    print()
    
    # Start the server
    try:
        # Change to project root directory
        subprocess.run(
            [sys.executable, str(server_script)],
            cwd=str(project_root),
            check=True
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n\nError: Server exited with code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())