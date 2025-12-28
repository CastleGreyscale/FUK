#!/usr/bin/env python3
"""
Quick verification that v1.2 is running correctly
"""

import requests
import sys

def check_backend():
    """Check if backend is running with v1.2"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        data = response.json()
        
        print("="*60)
        print("Backend Status")
        print("="*60)
        
        if data.get("version") == "1.2":
            print("✓ Backend version: 1.2")
        else:
            print(f"✗ Backend version: {data.get('version', 'unknown')} (expected 1.2)")
            print("  → Restart the backend: python fuk_web_server.py")
        
        features = data.get("features", [])
        expected_features = [
            "verbose_logging",
            "vram_clearing", 
            "time_tracking",
            "static_file_serving"
        ]
        
        for feature in expected_features:
            if feature in features:
                print(f"✓ {feature}")
            else:
                print(f"✗ {feature} - MISSING")
        
        print()
        return data.get("version") == "1.2"
        
    except requests.exceptions.ConnectionError:
        print("="*60)
        print("✗ Cannot connect to backend")
        print("="*60)
        print("\nBackend not running. Start it with:")
        print("  python fuk_web_server.py")
        print()
        return False
    except Exception as e:
        print(f"✗ Error checking backend: {e}")
        return False

def check_frontend():
    """Check if frontend is accessible"""
    try:
        response = requests.get("http://localhost:3000", timeout=2)
        
        print("="*60)
        print("Frontend Status")
        print("="*60)
        
        if "v1.2" in response.text:
            print("✓ Frontend version: 1.2")
            return True
        else:
            print("✗ Frontend version: NOT 1.2")
            print("  → Hard reload browser: Ctrl+Shift+R")
            print("  → Or close and reopen the tab")
            return False
            
    except requests.exceptions.ConnectionError:
        print("="*60)
        print("✗ Cannot connect to frontend")
        print("="*60)
        print("\nFrontend server not running. Start it with:")
        print("  python -m http.server 3000")
        print()
        return False
    except Exception as e:
        print(f"✗ Error checking frontend: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("FUK Web UI v1.2 - Verification Check")
    print("="*60 + "\n")
    
    backend_ok = check_backend()
    frontend_ok = check_frontend()
    
    print("="*60)
    print("Summary")
    print("="*60)
    
    if backend_ok and frontend_ok:
        print("✓ Everything looks good!")
        print("\nNext steps:")
        print("1. Open http://localhost:3000 in browser")
        print("2. Check for 'v1.2' in the header")
        print("3. Open browser console (F12)")
        print("4. Look for: '=== FUK UI v1.2 Loaded ==='")
        print()
    else:
        print("✗ Some checks failed")
        print("\nFollow the instructions above to fix")
        print()
    
    return 0 if (backend_ok and frontend_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
