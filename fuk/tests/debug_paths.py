#!/usr/bin/env python3
"""
Debug script to demonstrate the path issue
"""
from pathlib import Path
import os

print("=== Path Debugging ===\n")

# Show current working directory
print(f"Current working directory: {os.getcwd()}")
print(f"Current working directory (Path): {Path.cwd()}\n")

# Show what happens with relative paths
relative_path = Path("inputs/funtest_refimg01.png")
absolute_path = Path("inputs/funtest_refimg01.png").resolve()

print(f"Relative path: {relative_path}")
print(f"Relative exists? {relative_path.exists()}\n")

print(f"Absolute path: {absolute_path}")
print(f"Absolute exists? {absolute_path.exists()}\n")

# When you pass a relative path to musubi (which changes directory):
print("=== What Musubi Sees ===")
print(f"Musubi receives as string: '{str(relative_path)}'")
print("Musubi changes to vendor/musubi-tuner/")
print("Now 'inputs/funtest_refimg01.png' points to:")
print("  vendor/musubi-tuner/inputs/funtest_refimg01.png")
print("  ^ This doesn't exist!\n")

print("=== The Fix ===")
print(f"Pass absolute path instead: {absolute_path}")
print("This works from ANY directory")