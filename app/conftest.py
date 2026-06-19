import os
import faulthandler

def pytest_configure(config):
    """
    Executes instantly inside every single spawned parallel xdist worker process.
    Configures low-level system dump logs to expose the exact C++ file crash lines.
    """
    # 1. Arm environmental parameters safely
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["OMP_DYNAMIC"] = "FALSE"
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # os.environ["OMP_STACKSIZE"] = "64M"
    
    # 2. Assign an isolated log file for this specific process worker
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    log_path = f"worker_{worker_id}_crash.log"
    
    try:
        log_file = open(log_path, "w")
        # Direct raw C++ SIGSEGV dump streams into the file
        faulthandler.enable(file=log_file, all_threads=True)
    except Exception:
        pass