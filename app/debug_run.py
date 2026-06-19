# tests/debug_run.py
import sys
import os
import faulthandler

# 1. Enable the crash handler immediately at startup
faulthandler.enable()

# Ensure Python can find the test directory module paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import your async test scenario function
from app.tests.test_american_call import test_american_call_fine

def main():
    print("Opening the compiled dylib asset...")
    print("🚀 Running async C++ test directly via asyncio event loop...")
    
    # 2. Await the coroutine function to execute the calculation
    test_american_call_fine()
    
    print("✅ Finished safely without a segmentation fault.")

if __name__ == "__main__":
    main()