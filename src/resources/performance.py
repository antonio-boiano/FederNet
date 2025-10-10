# %%
import json
import os
import math
import psutil
import warnings
from typing import List, Dict, Any, Sequence, Union
import numpy as np

def generate_container_configs(client_count: int,
                                device_parma: Dict[str,int] = None,
                                host_score= 1150,
                                warn_level: int = 1,
                                cpu_period: int = 100_000) -> Dict[str, Any]:
    """
    Generate Docker HostConfig dicts for simulating devices or max host-capacity split.

    Args:
      client_count: Number of containers to create (>=1).
      device_type: Key in specs JSON (e.g., 'rpi4'); if None, split host resources.
      specs_path: Filesystem path to device_specs.json.
      cpu_period: CFS period in microseconds.

    Returns:
      Dict with 'HostConfig' for Docker API.
    """
    import os, psutil, json, warnings

    if client_count < 1:
        raise ValueError("client_count must be >= 1")

    host_cores = os.cpu_count() or 1
    host_ram_total = psutil.virtual_memory().total
    host_freq_info = psutil.cpu_freq()
    host_max_freq = host_freq_info.max or host_freq_info.current

    if device_parma:
        dev = device_parma 
        device_cores = dev["cores"]
        device_ram_bytes = dev["ram_gib"] * 1024**3
        device_freq = dev["freq_mhz"]
        device_score = dev.get("single_core_score", None)
        perf_ratio = device_score / host_score

        nano_cpus = int(device_cores * 1e9 * perf_ratio)
        cpu_quota = int(cpu_period * device_cores * perf_ratio)
        cpu_shares = int(device_cores * 1024)
        memory = int(device_ram_bytes)
        
        # Warn if device exceeds host
        total_requested_cores = nano_cpus * client_count/ 1e9
        total_requested_ram = device_ram_bytes * client_count
        
        if total_requested_cores > host_cores and warn_level > 0:
            warnings.warn(
                f"Requested {total_requested_cores} cores, "
                f"but host only has {host_cores} cores."
            )
        if total_requested_ram > host_ram_total and warn_level > 0:
            warnings.warn(
                f"Requested {total_requested_ram / 1024**3:.2f} GiB RAM, "
                f"but host only has {host_ram_total / 1024**3:.2f} GiB."
            )
        if device_freq > host_max_freq and warn_level > 0:
            warnings.warn(
                f"Requested {device_freq} MHz, "
                f"but host max CPU freq is {host_max_freq:.0f} MHz."
            )



    else:
        cores_per_container = host_cores / client_count
        ram_per_container = host_ram_total / client_count

        nano_cpus = int(cores_per_container * 1e9)
        cpu_quota = int(cpu_period * cores_per_container)
        cpu_shares = int(cores_per_container * 1024)
        memory = int(ram_per_container)

    config = {
        "NanoCPUs": nano_cpus,
        "Cpus": device_cores if device_parma else cores_per_container,
        "CpuPeriod": cpu_period,
        "CpuQuota": cpu_quota,
        "CpuShares": cpu_shares,
        "Memory": memory
    }

    return config


def load_profile_data(json_file):
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, json_file)
            return load_profile_data(json_path)
        except:
            print(f'I am in {os.getcwd()}')
            raise FileNotFoundError(f"File '{json_file}' not found.")

def extract_param_stats(param_dict):
    """
    Extract mean and std-dev for each parameter.
    If 'range' exists, use (max-min)/4 as std-dev (approx for normal covering 95%).
    If only 'typical' and no range, use typical and 20% as fallback stdev.
    """
    means, stddevs, keys = [], [], []
    for k, v in param_dict.items():
        if "typical" in v and "range" in v:
            mean = v["typical"]
            r = v["range"]
            # Empirical rule: (max-min)/4 ~ stdev for 95% of normal
            stddev = (r[1] - r[0]) / 4 if r[1] > r[0] else 0.2 * abs(mean)
        elif "typical" in v:
            mean = v["typical"]
            stddev = 0.2 * abs(mean)
        else:
            continue
        means.append(mean)
        stddevs.append(stddev)
        keys.append(k)
    return np.array(means), np.array(stddevs), keys

def deep_get(d: dict, key: str):
    """
    Depth-first search for *key* in a nested dict/list structure.
    Returns (value, path) on success; raises KeyError if not found.
    """
    stack = [(d, [])]                       # (current node, path_so_far)

    while stack:
        node, path = stack.pop()

        if isinstance(node, dict):
            if key in node:                 # found it at this level
                return node[key], path + [key]

            # search children
            for k, v in node.items():
                stack.append((v, path + [k]))

        elif isinstance(node, list):
            for i, item in enumerate(node):
                stack.append((item, path + [i]))

    raise KeyError(f"'{key}' not found anywhere in structure")

def sample_profile(
    data, 
    profile: str,
    idx: int,
    conn_type: str = None,
    corr: float = None
):
    """
    Given data loaded from JSON, sample a realistic set of parameters for a 
    given profile and optionally a specific connection type, using a 
    repeatable seed (idx).

    Returns dict mapping param name to sampled value.
    """
    # Select profile ("low", "middle", "high", "ultra")
    if profile not in data:
        _, full_prof = deep_get(data, profile)
        profile = full_prof[0]
        conn_type = full_prof[1]
        #raise ValueError(f"Profile '{profile}' not found in data")
    p_data = data[profile]
        
    if conn_type is None:
        # Average means and stddevs across all connection types
        all_means = []
        all_stds = []
        keys = None
        for ct in p_data:
            means, stds, these_keys = extract_param_stats(p_data[ct])
            if keys is None:
                keys = these_keys
            else:
                assert keys == these_keys, "Parameter keys must match across connection types"
            all_means.append(means)
            all_stds.append(stds)
        means = np.mean(all_means, axis=0)
        stddevs = np.mean(all_stds, axis=0)
    else:
        if conn_type not in p_data:
            raise ValueError(f"Connection type '{conn_type}' not in profile '{profile}'")
        means, stddevs, keys = extract_param_stats(p_data[conn_type])
    
    rng = np.random.RandomState(idx)
    if corr is not None:
        n = len(means)
        cov = np.full((n, n), corr, dtype=float)
        np.fill_diagonal(cov, 1.0)
        cov = cov * np.outer(stddevs, stddevs)        
        sample = rng.multivariate_normal(mean=means, cov=cov)
    else:
        sample = rng.normal(loc=means, scale=stddevs)
    # For integer metrics, round, for floats just use as is.
    result = {}
    for i, k in enumerate(keys):
        v = float(sample[i])
        v = max(0.0, v) # Negative delay or percent is not physical
        result[k] = v
    return result


def perturb_device(
    data: dict,
    device_name: str,
    idx: int,
    variation: float = 0.2,
    corr: float = 0.5,
    lock_keys: list = None
) -> dict:
    """
    Perturb the numeric specs of a given device using a multivariate normal distribution.
    
    Args:
      data: Dict[str, Dict] - Device data as provided.
      device_name: str - Device key in the data dict.
      idx: int - Seed for deterministic sampling.
      variation: float - Relative std-dev (20% means stddev=0.2*value).
      corr: float - Desired correlation between parameters.
      lock_keys: list of str - Keys that should NOT be perturbed.
    
    Returns:
      Dict[str, numeric]: Perturbed values, keys same as input.
    """
    if device_name not in data:
        raise ValueError(f"Device '{device_name}' not in data.")

    lock_keys = set(lock_keys or [])
    device = data[device_name]
    keys = [k for k in device if k not in lock_keys]
    locked = {k: device[k] for k in device if k in lock_keys}

    means = np.array([device[k] for k in keys], dtype=float)
    n = len(means)
    if n == 0:
        return dict(locked)

    sigma = variation * np.abs(means)
    cov = np.full((n, n), corr, dtype=float)
    np.fill_diagonal(cov, 1.0)
    cov = cov * np.outer(sigma, sigma)
    rng = np.random.RandomState(idx)
    sample = rng.multivariate_normal(mean=means, cov=cov)

    # Clamp negative values for physical parameters
    perturbed = {}
    for i, k in enumerate(keys):
        v = sample[i]
        if device[k] == int(device[k]):
            v = int(round(v))
        # All physical specs must be positive
        v = max(1, v)
        perturbed[k] = v

    perturbed.update(locked)
    return perturbed


def network_profile(
    profile:  Union[str, List[str]],
    idx: int,
    conn_type: str = None,
    corr: float = None
):
    if isinstance(profile, List):
        try:
            profile = profile[idx-1]
        except IndexError:
            profile = profile[-1]
    data = load_profile_data("network_specs.json")
    return sample_profile(data, profile, idx, conn_type, corr)
   

def device_profile(
    idx: int,
    host_score: int,
    device_name: Union[str, List[str]],
    client_count: int = 10,
    variation: float = 0.2,
    corr: float = 0.5,
    
):
    dev = None
    if device_name is not None:
        data = load_profile_data("device_specs.json")
        if isinstance(device_name, list):
            try:
                device_name = device_name[idx-1]
            except IndexError:
                if idx == 0:
                    return generate_container_configs(1, host_score=host_score)
                device_name = device_name[-1]
        dev =  perturb_device(data, device_name, idx, variation, corr, ["cores"])
    else:
        dev = None
    return generate_container_configs(client_count,device_parma=dev, host_score=host_score)


# # Example usage:
# if __name__ == "__main__":
#     data = load_profile_data("device_specs.json")
#     # Example: Don't perturb "cores"
#     a = perturb_device(data, "rpi4", idx=2, variation=0.3, lock_keys=["cores"])
#     print(generate_container_configs(10,a))
#     data = load_profile_data("network_specs.json")
#     # Sample for "high" profile, "wifi_80211ac" connection, idx=42
#     sample_profiled = sample_profile(data, "low", 2)
    





