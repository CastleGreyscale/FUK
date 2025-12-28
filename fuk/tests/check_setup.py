#!/usr/bin/env python3
"""
FUK Web UI - Project Structure Checker

Run this to verify your project is set up correctly.
"""

import sys
from pathlib import Path

def check_file(path, description, required=True):
    """Check if a file exists"""
    exists = path.exists()
    status = "✓" if exists else ("✗" if required else "⚠")
    print(f"{status} {description}")
    print(f"   {path}")
    if not exists and required:
        print(f"   ^ MISSING - Required!")
    elif not exists:
        print(f"   ^ Missing - Optional")
    print()
    return exists

def check_python_module(module_name):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError:
        print(f"✗ {module_name} - Not installed")
        return False

def main():
    print("\n" + "="*70)
    print("FUK Web UI - Project Structure Check")
    print("="*70 + "\n")
    
    base_dir = Path.cwd()
    print(f"Checking from: {base_dir}\n")
    
    # Check Python dependencies
    print("="*70)
    print("Python Dependencies")
    print("="*70 + "\n")
    
    deps_ok = True
    deps_ok &= check_python_module("fastapi")
    deps_ok &= check_python_module("uvicorn")
    
    if not deps_ok:
        print("\nInstall missing dependencies:")
        print("  pip install fastapi uvicorn python-multipart\n")
    print()
    
    # Check web UI files
    print("="*70)
    print("Web UI Files")
    print("="*70 + "\n")
    
    web_files_ok = True
    web_files_ok &= check_file(base_dir / "fuk_web_server.py", "Web Server", True)
    web_files_ok &= check_file(base_dir / "fuk_ui.jsx", "React UI", True)
    web_files_ok &= check_file(base_dir / "index.html", "HTML Host", True)
    web_files_ok &= check_file(base_dir / "start_web_ui.py", "Startup Script", False)
    
    # Check FUK core files
    print("="*70)
    print("FUK Core Files")
    print("="*70 + "\n")
    
    core_files_ok = True
    
    # Try different possible locations
    locations = [
        base_dir,
        base_dir / "core",
        base_dir / "fuk",
    ]
    
    found_location = None
    for loc in locations:
        if (loc / "musubi_wrapper.py").exists():
            found_location = loc
            break
    
    if found_location:
        print(f"Found FUK modules in: {found_location}\n")
        core_files_ok &= check_file(found_location / "musubi_wrapper.py", "Musubi Wrapper", True)
        core_files_ok &= check_file(found_location / "wan_video_wrapper.py", "Wan Video Wrapper", True)
        core_files_ok &= check_file(found_location / "generation_manager.py", "Generation Manager", True)
        core_files_ok &= check_file(found_location / "video_generation_manager.py", "Video Gen Manager", True)
        core_files_ok &= check_file(found_location / "format_convert.py", "Format Converter", True)
    else:
        print("✗ Cannot find FUK modules!")
        print("\nSearched in:")
        for loc in locations:
            print(f"  - {loc}")
        print("\nMake sure the following files exist:")
        print("  - musubi_wrapper.py")
        print("  - wan_video_wrapper.py")
        print("  - generation_manager.py")
        print("  - video_generation_manager.py")
        print("  - format_convert.py")
        print()
        core_files_ok = False
    
    # Check config files
    print("="*70)
    print("Configuration Files")
    print("="*70 + "\n")
    
    config_ok = True
    config_locations = [
        base_dir / "config",
        base_dir.parent / "config",
    ]
    
    found_config = None
    for loc in config_locations:
        if (loc / "models.json").exists():
            found_config = loc
            break
    
    if found_config:
        print(f"Found config in: {found_config}\n")
        config_ok &= check_file(found_config / "models.json", "Models Config", True)
        config_ok &= check_file(found_config / "defaults.json", "Defaults Config", True)
    else:
        print("✗ Cannot find config directory!")
        print("\nSearched in:")
        for loc in config_locations:
            print(f"  - {loc}")
        print()
        config_ok = False
    
    # Check musubi-tuner
    print("="*70)
    print("Musubi-Tuner")
    print("="*70 + "\n")
    
    musubi_locations = [
        base_dir / "vendor" / "musubi-tuner",
        base_dir.parent / "vendor" / "musubi-tuner",
    ]
    
    musubi_ok = False
    for loc in musubi_locations:
        if loc.exists():
            print(f"✓ Found musubi-tuner: {loc}\n")
            musubi_ok = True
            break
    
    if not musubi_ok:
        print("✗ Cannot find musubi-tuner!")
        print("\nSearched in:")
        for loc in musubi_locations:
            print(f"  - {loc}")
        print()
    
    # Output directory
    print("="*70)
    print("Output Directory")
    print("="*70 + "\n")
    
    output_locations = [
        base_dir / "outputs",
        base_dir.parent / "outputs",
    ]
    
    output_ok = False
    for loc in output_locations:
        if loc.exists():
            print(f"✓ Found outputs: {loc}\n")
            output_ok = True
            break
    
    if not output_ok:
        print("⚠ Output directory doesn't exist (will be created)")
        print(f"  Will create: {output_locations[0]}\n")
    
    # Summary
    print("="*70)
    print("Summary")
    print("="*70 + "\n")
    
    all_ok = deps_ok and web_files_ok and core_files_ok and config_ok and musubi_ok
    
    if all_ok:
        print("✓ All checks passed! You're ready to start the web UI.\n")
        print("Run:")
        print("  python fuk_web_server.py")
        print("\nOr use the all-in-one startup:")
        print("  python start_web_ui.py")
    else:
        print("✗ Some checks failed. Please fix the issues above.\n")
        
        if not deps_ok:
            print("Install dependencies:")
            print("  pip install fastapi uvicorn python-multipart\n")
        
        if not web_files_ok:
            print("Missing web UI files - make sure you copied all the files.")
        
        if not core_files_ok:
            print("Missing FUK core files - ensure you're in the project directory.")
        
        if not config_ok:
            print("Missing config files - check config/models.json and config/defaults.json exist.")
        
        if not musubi_ok:
            print("Missing musubi-tuner - ensure vendor/musubi-tuner exists.")
    
    print("="*70 + "\n")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
