import subprocess
import time
import re
import json
import threading
import tempfile
import os
from pathlib import Path

def run_single_instance(lp_file: Path, dec_file: Path, dual_file: Path, gcg_executable: Path, cmd_string: str, timeout: int, log_file: Path, save_logs: bool = True, print_solver_output: bool = False) -> tuple[str, float, float]:
    """
    Runs a single GCG instance using a provided SCIP command string.
    
    Returns:
        (str, dict): A tuple containing the status ("SUCCESS", "TIMEOUT", "MISSING_FILES", "CRASH")
                     and a metrics dictionary with solving_time, final_obj_val, and other metrics.
    """
    # Ensure all required files exist before running
    if not lp_file.exists():
        print(f"Error: Missing LP file at {lp_file}")
        return "MISSING_FILES", 0.0, float('inf'), 0
    if not dec_file.exists():
        print(f"Error: Missing DEC file at {dec_file}")
        return "MISSING_FILES", 0.0, float('inf'), 0
    if not dual_file.exists():
        print(f"Error: Missing DUAL file at {dual_file}")
        return "MISSING_FILES", 0.0, float('inf'), 0
        
    # Ensure the directory for the log file exists
    if save_logs:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate a unique temporary JSON file for GCG to write metrics into.
    # We close the fd immediately — GCG (C++) will be the one writing to it.
    metrics_fd, metrics_path = tempfile.mkstemp(suffix=".json", prefix="gcg_metrics_")
    os.close(metrics_fd)

    try:
        # Prepend the metrics_file setting so GCG knows where to write the JSON.
        full_cmd_string = (
            f'set pricingcb initduals metrics_file "{metrics_path}"\n'
            + cmd_string
        )

        with tempfile.NamedTemporaryFile(suffix=".batch", mode='w', delete=True) as temp_file:
            temp_file.write(full_cmd_string)
            temp_file.flush()
            batch_file_path = temp_file.name

            start_time = time.perf_counter()
            process = subprocess.Popen(
                [str(gcg_executable), "-b", batch_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, # Prevents hanging if SCIP drops to interactive shell
                text=True
            )
                    
            output_lines = []
            def reader_thread(pipe):
                for line in pipe:
                    output_lines.append(line)
                    if print_solver_output:
                        # flush=True forces Python to immediately print the line to the console
                        print(line, end="", flush=True)
                        
            if print_solver_output:
                print(f"\n--- SCIP OUTPUT START ({lp_file.name}) ---")

            t = threading.Thread(target=reader_thread, args=(process.stdout,))
            t.start()

            timeout_expired = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timeout_expired = True
                process.kill()
                process.wait()

            t.join() # Wait for the reader thread to finish
            real_runtime = time.perf_counter() - start_time
            full_output = "".join(output_lines)

            if save_logs:
                with open(log_file, "w") as f:
                    f.write(full_output)

            if timeout_expired:
                if print_solver_output:
                    print(f"\n--- SCIP OUTPUT TIMED OUT ({lp_file.name}) ---\n")
                return "TIMEOUT", float(timeout), float('inf'), 0
                
            if print_solver_output:
                print(f"\n--- SCIP OUTPUT END ({lp_file.name}) ---\n")

            # --- solving_time: read from GCG-written JSON, fallback to real_runtime ---
            solving_time = real_runtime
            try:
                with open(metrics_path, "r") as mf:
                    gcg_metrics = json.load(mf)
                solving_time = float(gcg_metrics["solving_time"])
            except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
                # File missing (GCG crashed before writing) or malformed — use wall-clock time
                pass

            # Dummy returns for all regex-based extractions while they are being migrated to JSON
            final_obj_val = float('inf')
            cols_needed = 0
            slp_iters_main = 0
            slp_iters_custom = 0

            status = "SUCCESS" if process.returncode == 0 else "CRASH"

            metrics = {
                "final_obj_val": final_obj_val,
                "solving_time": solving_time,
                "cols_needed_for_rmp_feasibility": cols_needed,
                "slp_iterations_main_loop": slp_iters_main,
                "slp_iterations_custom_pricing": slp_iters_custom
            }

            return status, metrics

    except Exception as e:
        print(f"Error executing SCIP process: {e}")
        return "CRASH", float(timeout), float('inf'), 0

    finally:
        # Always clean up the temporary metrics file
        try:
            os.remove(metrics_path)
        except FileNotFoundError:
            pass