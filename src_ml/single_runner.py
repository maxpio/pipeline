"""
Executes single instances of GCG and parses their outputs.
"""
import subprocess
import time
import re
import json
import threading
import tempfile
import os
from pathlib import Path

def run_single_instance(lp_file: Path, dec_file: Path, dual_file: Path, gcg_executable: Path, cmd_string: str, timeout: int, log_file: Path, save_logs: bool = True, print_solver_output: bool = False) -> tuple[str, dict]:
    """Runs a single GCG instance and extracts performance metrics."""
    # Check files
    if not lp_file.exists():
        print(f"Error: Missing LP file at {lp_file}")
        return "MISSING_FILES", {"final_dual_bound": float('inf'), "solving_time": 0.0}
    if not dec_file.exists():
        print(f"Error: Missing DEC file at {dec_file}")
        return "MISSING_FILES", 0.0, float('inf'), 0
    if not dual_file.exists():
        print(f"Error: Missing DUAL file at {dual_file}")
        return "MISSING_FILES", 0.0, float('inf'), 0
        
    # Check log dir
    if save_logs:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create temp metrics file
    metrics_fd, metrics_path = tempfile.mkstemp(suffix=".json", prefix="gcg_metrics_")
    os.close(metrics_fd)

    try:
        # Prepend metrics path
        full_cmd_string = (
            f'set pricingcb initduals metrics_file {metrics_path}\n'
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
                stdin=subprocess.DEVNULL, # Prevent hanging
                text=True
            )
                    
            output_lines = []
            def reader_thread(pipe):
                """Reads and optionally prints stdout."""
                for line in pipe:
                    output_lines.append(line)
                    if print_solver_output:
                        # Flush output
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

            t.join() # Wait for reader
            real_runtime = time.perf_counter() - start_time
            full_output = "".join(output_lines)

            if save_logs:
                with open(log_file, "w") as f:
                    f.write(full_output)

            if timeout_expired:
                if print_solver_output:
                    print(f"\n--- SCIP OUTPUT TIMED OUT ({lp_file.name}) ---\n")
                return "TIMEOUT", {"final_dual_bound": float('inf'), "solving_time": float(timeout)}
                
            if print_solver_output:
                print(f"\n--- SCIP OUTPUT END ({lp_file.name}) ---\n")

            # Fetch metrics
            try:
                with open(metrics_path, "r") as mf:
                    gcg_metrics = json.load(mf)
                
                solving_time = float(gcg_metrics.get("solving_time", real_runtime))
                
                final_dual_bound = gcg_metrics.get("final_dual_bound")
                if final_dual_bound is None:
                    final_dual_bound = float('inf')
                else:
                    final_dual_bound = float(final_dual_bound)
                    
                final_primal_bound = gcg_metrics.get("final_primal_bound")
                if final_primal_bound is None:
                    final_primal_bound = float('inf')
                else:
                    final_primal_bound = float(final_primal_bound)
                    
                cols_needed = int(gcg_metrics.get("cols_needed_for_rmp_feasibility", 0))
                slp_iters_main = int(gcg_metrics.get("slp_iterations_main_loop", 0))
                slp_iters_custom = int(gcg_metrics.get("slp_iterations_custom_pricing", 0))

            except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
                # Fallback on crash
                print(f"  [!] Warning: could not read GCG metrics JSON ({e}). Bounds will be reported as inf.")
                solving_time = real_runtime
                final_dual_bound = float('inf')
                final_primal_bound = float('inf')
                cols_needed = 0
                slp_iters_main = 0
                slp_iters_custom = 0

            # Parse stdout
            if final_dual_bound == float('inf'):
                m = re.search(r"Dual Bound\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?|[+-]?inf(?:inite|inity)?)", full_output, re.IGNORECASE)
                if m:
                    try:
                        val_str = m.group(1).lower()
                        if 'inf' in val_str:
                            final_dual_bound = float('inf') if '-' not in val_str else float('-inf')
                        else:
                            final_dual_bound = float(val_str)
                    except ValueError:
                        pass
            
            if final_primal_bound == float('inf'):
                m = re.search(r"Primal Bound\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?|[+-]?inf(?:inite|inity)?)", full_output, re.IGNORECASE)
                if m:
                    try:
                        val_str = m.group(1).lower()
                        if 'inf' in val_str:
                            final_primal_bound = float('inf') if '-' not in val_str else float('-inf')
                        else:
                            final_primal_bound = float(val_str)
                    except ValueError:
                        pass

            status = "SUCCESS" if process.returncode == 0 else "CRASH"

            metrics = {
                "final_dual_bound": final_dual_bound,
                "final_primal_bound": final_primal_bound,
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
        # Cleanup
        Path(metrics_path).unlink(missing_ok=True)