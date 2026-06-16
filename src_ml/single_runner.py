import subprocess
import time
import re
import threading
from pathlib import Path

def run_single_instance(lp_file: Path, dec_file: Path, dual_file: Path, gcg_executable: Path, cmd_string: str, timeout: int, log_file: Path, save_logs: bool = True, print_solver_output: bool = False) -> tuple[str, float, float]:
    """
    Runs a single GCG instance using a provided SCIP command string.
    
    Returns:
        (str, float, float, int): A tuple containing the status ("SUCCESS", "TIMEOUT", "MISSING_FILES", "CRASH"), 
                             the execution time in seconds,
                             the best objective value found (Primal Bound), and the total MLP iterations.
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

    try:
        import tempfile
        import os
        fd, batch_file_path = tempfile.mkstemp(suffix=".batch", text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(cmd_string)

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

        # Parse SCIP's reported solving time, fallback to real execution time if it fails.
        # We use findall and take the last match to avoid capturing intermediate subproblem times.
        matches = re.findall(r"Solving Time\s*\(sec\)\s*:\s*([\d\.]+)", full_output)
        execution_time = float(matches[-1]) if matches else real_runtime

        # Parse SCIP's Primal Bound (Objective Value)
        pb_matches = re.findall(r"Primal Bound\s*:\s*([\+\-a-zA-Z0-9\.]+)", full_output)
        objective_value = float('inf')
        if pb_matches:
            val_str = pb_matches[-1]
            if val_str not in ('-', 'infinity'):
                try:
                    objective_value = float(val_str)
                except ValueError:
                    pass

        # Parse MLP iterations
        # This regex matches the line layout: "   time | node | left | SLP iter | MLP iter |"
        mlp_matches = re.findall(r"^.*?(?:[\d\.]+[smh])\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|", full_output, re.MULTILINE)
        mlp_iters = int(mlp_matches[-1]) if mlp_matches else 0

        status = "SUCCESS" if process.returncode == 0 else "CRASH"
        if os.path.exists(batch_file_path):
            os.remove(batch_file_path)
        return status, execution_time, objective_value, mlp_iters

    except Exception as e:
        print(f"Error executing SCIP process: {e}")
        if 'batch_file_path' in locals() and os.path.exists(batch_file_path):
            os.remove(batch_file_path)
        return "CRASH", float(timeout), float('inf'), 0