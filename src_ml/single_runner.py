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

    import tempfile
    import os
    try:
        with tempfile.NamedTemporaryFile(suffix=".batch", mode='w', delete=True) as temp_file:
            temp_file.write(cmd_string)
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

            # Parse SCIP's reported solving time, fallback to real execution time if it fails.
            matches = re.findall(r"Solving Time\s*\(sec\)\s*:\s*([\d\.]+)", full_output)
            solving_time = float(matches[-1]) if matches else real_runtime

            # 1. Dual Bound (final obj val)
            db_matches = re.findall(r"Dual Bound\s*:\s*([\+\-a-zA-Z0-9\.]+)", full_output)
            final_obj_val = float('inf')
            if db_matches:
                val_str = db_matches[-1]
                if val_str not in ('-', 'infinity'):
                    try:
                        final_obj_val = float(val_str)
                    except ValueError:
                        pass

            # 2. Columns needed for feasibility of RMP
            pre_pricing_text = full_output.split("Starting reduced cost pricing...")[0] if "Starting reduced cost pricing..." in full_output else full_output
            mvars_pattern = r"^[^\n]*?(?:[\d\.]+[smh])\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[^\s]+\s*\|\s*[^\s]+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|"
            mvars_matches = re.findall(mvars_pattern, pre_pricing_text, re.MULTILINE)
            cols_needed = int(mvars_matches[-1]) if mvars_matches else 0

            # 3. SLP iterations main loop
            slp_pattern = r"^[^\n]*?(?:[\d\.]+[smh])\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|"
            slp_matches = re.findall(slp_pattern, full_output, re.MULTILINE)
            slp_iters_main = int(slp_matches[-1]) if slp_matches else 0

            # 4. SLP iterations custom pricing
            custom_slp_match = re.search(r"Total simplex iterations in custom pricing loop:\s*(\d+)", full_output)
            slp_iters_custom = int(custom_slp_match.group(1)) if custom_slp_match else 0

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