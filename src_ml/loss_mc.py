"""
PyTorch autograd function for batched subgradient loss using parallel SCIP solvers.
"""
import os
import yaml
import torch
import time
from collections import OrderedDict
from src_ml.lagrangian_relaxation import LagrangianRelaxation

# Load config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config_ml.yaml")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# SCIP settings
DISABLE_PRESOLVE = config['training_parameters']['scip_settings']['disable_presolve']
DISABLE_HEURISTICS = config['training_parameters']['scip_settings']['disable_heuristics']
MIP_GAP = config['training_parameters']['scip_settings'].get('mip_gap', 0.0)

# Worker init and solve
class LRUScipCache:
    """LRU cache for SCIP environments."""
    def __init__(self, capacity=50):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            oldest_key, oldest_env = self.cache.popitem(last=False)
            try:
                oldest_env.model.freeProb()
            except Exception:
                pass

class WorkerState:
    """Process-local namespace for workers."""
    lag_relax_cache = None
    cache_enabled = None

def worker_init(cache_flag, capacity):
    """Initializes the local cache on worker."""
    WorkerState.cache_enabled = cache_flag
    if cache_flag:
        WorkerState.lag_relax_cache = LRUScipCache(capacity=capacity)
    else:
        WorkerState.lag_relax_cache = None

def worker_solve(file_name, lp_path, mult_dict, mip_gap):
    """Executes SCIP optimization."""
    start_time = time.time()
    
    if WorkerState.cache_enabled and file_name in WorkerState.lag_relax_cache:
        lag_relax_env = WorkerState.lag_relax_cache[file_name]
        lag_relax_env.model.setRealParam("limits/gap", mip_gap)
    else:
        lag_relax_env = LagrangianRelaxation(
            lp_path, 
            disable_presolve=DISABLE_PRESOLVE, 
            disable_heuristics=DISABLE_HEURISTICS,
            mip_gap=mip_gap
        )
        if WorkerState.cache_enabled:
            WorkerState.lag_relax_cache[file_name] = lag_relax_env
            
    lag_relax_env.set_multipliers(mult_dict)
    bound, violations_dict = lag_relax_env.optimize_and_get_violations()
    elapsed = time.time() - start_time
    return bound, violations_dict, elapsed

# Batched subgradient loss
class BatchedSubgradientLoss(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, lambda_raw_batch, eq_flags_batch, batch_sizes, worker_args_list, executors, file_to_worker_map, mip_gap):
        """Computes loss and subgradients."""
        # Enforce non-negativity
        actual_multipliers = torch.where(eq_flags_batch > 0.5, lambda_raw_batch, torch.relu(lambda_raw_batch))
        
        # Pull to CPU
        actual_multipliers_cpu = actual_multipliers.detach().cpu()
        
        # Split multipliers
        actual_mult_splits = torch.split(actual_multipliers_cpu, batch_sizes)
        
        # Call SCIP solvers
        futures = []
        for i, (file_name, lp_path, cons_names) in enumerate(worker_args_list):
            mult_dict = dict(zip(cons_names, actual_mult_splits[i].tolist()))
            worker_id = file_to_worker_map.get(file_name, 0)
            executor = executors[worker_id] if isinstance(executors, list) else executors
            futures.append(executor.submit(worker_solve, file_name, lp_path, mult_dict, mip_gap))
            
        bounds = []
        violations_list = []
        solve_times = []
        for future in futures:
            bound, violations_dict, elapsed = future.result()
            if bound is None:
                raise RuntimeError("SCIP failed to find an optimal solution in worker.")
            bounds.append(bound)
            violations_list.append(violations_dict)
            solve_times.append(elapsed)
            
        # Extract subgradients
        all_subgradients = []
        for i, (file_name, lp_path, cons_names) in enumerate(worker_args_list):
            v_dict = violations_list[i]
            all_subgradients.extend([v_dict[c_name] for c_name in cons_names])
            
        # Transfer to GPU
        subgrad_tensor = torch.tensor(all_subgradients, dtype=lambda_raw_batch.dtype).pin_memory().to(lambda_raw_batch.device, non_blocking=True)
            
        # Save for backward
        ctx.save_for_backward(subgrad_tensor, actual_multipliers, eq_flags_batch)
        
        # Sum bounds
        loss = torch.tensor([sum(bounds)], dtype=lambda_raw_batch.dtype, device=lambda_raw_batch.device)
        return loss, bounds, solve_times

    @staticmethod
    def backward(ctx, grad_loss, grad_bounds, grad_solve_times=None):
        """Computes gradients for multipliers."""
        subgrad_tensor, actual_multipliers, eq_flags_batch = ctx.saved_tensors
        
        # Math gradient
        grad_multipliers = -subgrad_tensor * grad_loss
        
        # KKT fix
        zero_grad_mask = (eq_flags_batch < 0.5) & (actual_multipliers <= 1e-6) & (subgrad_tensor < 0)
        grad_multipliers[zero_grad_mask] = 0.0
        
        # Match inputs
        return grad_multipliers, None, None, None, None, None, None